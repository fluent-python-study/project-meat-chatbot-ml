import os
import sys
import argparse

from transformers import *
from summarizer import Summarizer
from kobert_transformers import get_tokenizer

from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='test.csv')
    parser.add_argument('--bname', type=str, default='monologg/kobert')
    args = parser.parse_args()
    return args


class BertExtractiveSummarizer(object):
    def __init__(self, bert_name):
        self._load_custom_model(bert_name)
        
    def _load_custom_model(self, bert_name):
        custom_config = AutoConfig.from_pretrained(bert_name)    
        custom_config.output_hidden_states = True
        custom_tokenizer = get_tokenizer()
        custom_model = AutoModel.from_pretrained(bert_name, config=custom_config)
        
        self.model = Summarizer(custom_model=custom_model,
                custom_tokenizer=custom_tokenizer)

    def summarize(self, input_lst, **kwargs):
        chats = '\n'.join(input_lst)
        summary = self.model(chats, min_length=20, num_sentences=3)
        return summary


def main():
    args = get_args()
    chats = load_kakao_csv(args.fname)
    
    summarizer = BertExtractiveSummarizer(bert_name=args.bname)
    print('Origin')
    print(chats)

    print('Summarize')
    print(summarizer.summarize(chats))

if __name__ == '__main__':
    main()
