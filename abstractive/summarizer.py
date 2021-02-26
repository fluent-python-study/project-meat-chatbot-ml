# Transformer based Text summarization model
import os
import sys
import re
import math
import random
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import Sequence, NFKC, Replace
from abstractive.network import *
from utils import load_json, set_seed


UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data/samsum_corpus')
    parser.add_argument('--src_key', type=str, default='dialogue')
    parser.add_argument('--tgt_key', type=str, default='summary')
    parser.add_argument('--res_prefix', type=str, default='samsum')
    ## tokenizer
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--tokenizer_dir', type=str, default='./samsum_tokenizer')
    ## model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_encoder', type=int, default=6)
    parser.add_argument('--n_decoder', type=int, default=6)
    parser.add_argument('--intermediate_size', type=int, default=1024)
    ## training
    parser.add_argument('--max_len', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_iter', type=int, default=10)
    parser.add_argument('--inf_batch_size', type=int, default=8)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--adam_betas', type=str, default='(0.9, 0.98)')
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--num_warmup_steps', type=int, default=4000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)

    args = parser.parse_args()
    return args


def get_tokenizer(args):
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = Sequence([
        NFKC(),
        Replace('\r', ''),
        Replace('\n', ' ')])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    
    if os.path.isdir(args.tokenizer_dir):
        vocab_fn = os.path.join(args.tokenizer_dir, 'vocab.json')
        merge_fn = os.path.join(args.tokenizer_dir, 'merges.txt')
        tokenizer.model = models.BPE.from_file(vocab_fn, merge_fn)
    else:
        os.makedirs(args.tokenizer_dir)
        trainer = trainers.BpeTrainer(
                vocab_size=args.vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
                )
        files = [ os.path.join(args.data_dir, split) for split in ['train.json', 'val.json', 'test.json']]
        tokenizer.train(files=files, trainer=trainer)
        tokenizer.model.save(args.tokenizer_dir)
        
    return tokenizer


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(data, tokenizer, src_key, tgt_key, max_seq_length=None):
    src_lst = [x[src_key] for x in data]
    src_token_ids = []
    for src in src_lst:
        src_ids = [BOS_ID] + tokenizer.encode(src).ids + [EOS_ID]
        src_token_ids.append(src_ids)
    
    tgt_lst = [x[tgt_key] for x in data]
    tgt_token_ids = []
    for tgt in tgt_lst:
        tgt_ids = [BOS_ID] + tokenizer.encode(tgt).ids + [EOS_ID]
        tgt_token_ids.append(tgt_ids)
    
    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) for x in tgt_token_ids])
    if max_seq_length and max_seq_length < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [PAD_ID] * src_pad_length)
        src_padding_mask.append([0] * len(src) + [1] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [PAD_ID] * tgt_pad_length)
        tgt_padding_mask.append([0] * (len(tgt) - 1) + [1] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool()
    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]


class Trainer(object):
    def __init__(self, args, trains, vals):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.args = args
        self.tokenizer = get_tokenizer(args)
        self.trainset = TextDataset(trains)
        self.valset = TextDataset(vals)

        self.t_loader = DataLoader(self.trainset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: collate_fn(x, self.tokenizer,
                    args.src_key, args.tgt_key, args.max_len))
        
        self.v_loader = DataLoader(self.valset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: collate_fn(x, self.tokenizer,
                    args.src_key, args.tgt_key, args.max_len))
        
        self.model = TransformerModel(vocab_size=args.vocab_size,
                d_model=args.d_model,
                num_attention_heads=args.n_heads,
                num_encoder_layers=args.n_encoder,
                num_decoder_layers=args.n_decoder,
                intermediate_size=args.intermediate_size,
                max_len=args.max_len)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                betas=eval(args.adam_betas), eps=args.eps,
                weight_decay=args.weight_decay)
        lr_lambda = lambda x:x/args.num_warmup_steps if x <= args.num_warmup_steps else (x/args.num_warmup_steps) ** -0.5
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def save_checkpoint(self, epoch, checkpoint_fname):
        torch.save({
            'epoch':epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
            }, checkpoint_fname)

    def load_checkpoint(self, checkpoint_fname):
        checkpoint = torch.load(checkpoint_fname)
        self.model.loaad_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def run_batch(self, batch, train):
        batch = tuple(t.to(self.device) for t in batch)
        src, tgt, src_mask, tgt_mask, tgt_label = batch

        output = self.model(src, tgt, src_mask, tgt_mask)
        batch_size = tgt.size(1)
        raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt_label.view(-1), reduction='none')
        raw_loss = raw_loss.view(-1, batch_size)
        loss = (raw_loss * torch.logical_not(tgt_mask).t().float()).sum(0).mean()
        if train:
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
            self.scheduler.step()
        items = [loss.data.item(), batch_size, tgt_mask.sum().item()]
        return items

    def train(self, epoch=None):
        self.model.train()
        tot_loss = 0.
        cnt_sent = 0
        cnt_token = 0
        
        tqdm_batch_iterator = tqdm(self.t_loader)
        for batch_idx, batch in enumerate(tqdm_batch_iterator):
            loss, n_sent, n_token = self.run_batch(batch, train=True)
            tot_loss += loss * n_sent
            cnt_sent += n_sent
            cnt_token += n_token

            description = "[Epoch: {:3d}][Loss: {:6f}][lr : {:7f}]".format(epoch,
                    loss,
                    self.optimizer.param_groups[0]['lr'])
            tqdm_batch_iterator.set_description(description)
        status = {'loss_sent' : tot_loss / cnt_sent,
                'token_ppl' : math.exp(tot_loss/cnt_token)}
        return status
    
    def valid(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            tot_loss = 0.
            cnt_sent = 0
            cnt_token = 0
            for batch_idx, batch in enumerate(self.v_loader):
                loss, n_sent, n_token = self.run_batch(batch, train=False)
                tot_loss += loss * n_sent
                cnt_sent += n_sent
                cnt_token += n_token

            status = {'loss_sent' : tot_loss / cnt_sent,
                    'token_ppl' : math.exp(tot_loss/cnt_token)}
        return status
    
    def summarize(self, origins):
        self.model.eval()
        prediction = []
        with torch.no_grad():
            for i in tqdm(range(0, len(origins), self.args.inf_batch_size)):
                batch = origins[i:i+self.args.inf_batch_size]
    
                src_token_ids = []
                for src in batch:
                    src_ids = [BOS_ID] + tokenizer.encode(src).ids + [EOS_ID]
                    src_token_ids.append(src_ids)
                
                src_seq_length = [len(x) for x in src_token_ids]
                src_max_seq_length = max(src_seq_length)
                if self.args.max_len < src_max_seq_length:
                    src_max_seq_length = self.args.max_len
                src_padded = []
                src_padding_mask = []
                for x in src_token_ids:
                    x = x[:src_max_seq_length]
                    src_pad_length = src_max_seq_length - len(x)
                    src_padded.append(x + [PAD_ID] * src_pad_length)
                    src_padding_mask.append([0] * len(x) + [1] * src_pad_length)
                src_padded = torch.tensor(src_padded).t().contiguous().to(self.device)
                src_padding_mask = torch.tensor(src_padding_mask).bool().to(self.device)
                
                memory = self.model.encode(src_padded, src_padding_mask)
                
                tgt_token_ids = [[BOS_ID] for _ in batch]
                end = [False for _ in batch]
                for l in range(src_max_seq_length):
                    tgt = torch.tensor(tgt_token_ids).t().contiguous().to(self.device)
                    output = self.model.decode(tgt, memory, memory_key_padding_mask=src_padding_mask)
                    top1 = output[-1].argmax(-1).tolist()
                    for i, tok in enumerate(top1):
                        if tok == EOS_ID or l >= src_seq_length[i]:
                            end[i] = True
                        tgt_token_ids[i].append(tok if not end[i] else EOS_ID)
                    if all(end):
                        break
                prediction.extend([self.sp.Decode(tgt) for tgt in tgt_token_ids])
        return prediction

    def run(self):
        train_report = 'train loss sent : {:.4f}, train token ppl : {:.4f}'
        val_report = 'val loss sent : {:.4f}, val token ppl : {:.4f}'

        for epoch in range(1, self.args.epochs + 1):
            epoch_status = self.train(epoch)
            print(train_report.format(
                epoch_status['loss_sent'],
                epoch_status['token_ppl']))
            if epoch % self.args.val_iter == 0:
                valid_status = self.valid(epoch)
                print(val_report.format(
                    valid_status['loss_sent'],
                    valid_status['token_ppl']))
            if epoch % 20 == 0:
                self.save_checkpoint(epoch, '{}_{}_{}.cp'.format(self.args.res_prefix,
                    self.args.vocab_size,
                    epoch)) 
        
        self.save_checkpoint(epoch, '{}_{}_{}.cp'.format(self.args.res_prefix,
            self.args.vocab_size,
            epoch)) 


class Summarizer(object):
    def __init__(self, tokenizer_dir,
            vocab_size,
            d_model,
            n_heads,
            n_encoder,
            n_decoder,
            intermediate_size,
            max_len,
            batch_size,
            checkpoint_fname):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = self.get_tokenizer(tokenizer_dir)
        self.batch_size = batch_size
        self.max_len = max_len
        self.model = TransformerModel(vocab_size=vocab_size,
                d_model=d_model,
                num_attention_heads=n_heads,
                num_encoder_layers=n_encoder,
                num_decoder_layers=n_decoder,
                intermediate_size=intermediate_size,
                max_len=max_len)
        self.model.to(self.device)
        self.load_checkpoint(checkpoint_fname)
        
    def get_tokenizer(self, tokenizer_dir):
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = Sequence([
            NFKC(),
            Replace('\r', ''),
            Replace('\n', ' ')])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        
        vocab_fn = os.path.join(tokenizer_dir, 'vocab.json')
        merge_fn = os.path.join(tokenizer_dir, 'merges.txt')
        tokenizer.model = models.BPE.from_file(vocab_fn, merge_fn)
        tokenizer.add_special_tokens(['[UNK]', '[PAD]', '[BOS]', '[EOS]'])
        return tokenizer

    def load_checkpoint(self, checkpoint_fname):
        checkpoint = torch.load(checkpoint_fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def summarize(self, origins):
        self.model.eval()
        prediction = []
        with torch.no_grad():
            for i in tqdm(range(0, len(origins), self.batch_size)):
                batch = origins[i:i+self.batch_size]
    
                src_token_ids = []
                for src in batch:
                    src_ids = [BOS_ID] + self.tokenizer.encode(src).ids + [EOS_ID]
                    src_token_ids.append(src_ids)
                
                src_seq_length = [len(x) for x in src_token_ids]
                src_max_seq_length = max(src_seq_length)
                if self.max_len < src_max_seq_length:
                    src_max_seq_length = self.max_len
                src_padded = []
                src_padding_mask = []
                for x in src_token_ids:
                    x = x[:src_max_seq_length]
                    src_pad_length = src_max_seq_length - len(x)
                    src_padded.append(x + [PAD_ID] * src_pad_length)
                    src_padding_mask.append([0] * len(x) + [1] * src_pad_length)
                src_padded = torch.tensor(src_padded).t().contiguous().to(self.device)
                src_padding_mask = torch.tensor(src_padding_mask).bool().to(self.device)
                
                memory = self.model.encode(src_padded, src_padding_mask)
                
                tgt_token_ids = [[BOS_ID] for _ in batch]
                end = [False for _ in batch]
                for l in range(src_max_seq_length):
                    tgt = torch.tensor(tgt_token_ids).t().contiguous().to(self.device)
                    output = self.model.decode(tgt, memory, memory_key_padding_mask=src_padding_mask)
                    top1 = output[-1].argmax(-1).tolist()
                    for i, tok in enumerate(top1):
                        if tok == EOS_ID or l >= src_seq_length[i]:
                            end[i] = True
                        tgt_token_ids[i].append(tok if not end[i] else EOS_ID)
                    if all(end):
                        break
                prediction.extend([self.tokenizer.decode(tgt, skip_special_tokens=True) for tgt in tgt_token_ids])
        return prediction

def main():
    args = get_args()
    set_seed(args.seed)
    trains = load_json(os.path.join(args.data_dir, 'train.json'))
    vals = load_json(os.path.join(args.data_dir, 'val.json'))
    trainer = Trainer(args, trains, vals)
    trainer.run()
    
#    
#    orginal_tests = [test['article_original'] for test in tests]
#    predicts = trainer.summarize(original_tests)
#    results = []
#    for pred, test in zip(predicts, tests):
#        summary = '{}\n{}\n{}'
#        outputs = extractor.extract(pred, test['article_original'])
#        result = {
#                'id':test['id'],
#                'summary':summary.format(*outputs)
#                }
#    df = pd.DataFrame(results)
#    df.to_csv('result.csv')

if __name__ == '__main__':
    main()
