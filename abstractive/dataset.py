# dataset
import torch.utils.data import Dataset

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(data, tokenizer, src_key, tgt_key, max_seq_length=None):
    src_sent = [x[src_key] for x in data]
    src_token_ids = []
    for sent_lst in src_sent:
        src_sequence = ''
        for sent in sent_lst:
            src_sequence += sent + ' '
        src_token_ids.append(tokenizer.encode(src_sequence.rstrip(), add_special_tokens=True))
    tgt_token_ids = [tokenizer.encode(x[tgt_key], add_special_tokens=True) for x in data]
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
