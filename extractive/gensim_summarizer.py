import os
import sys
import argparse

from gensim.summarization import summarize

from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='test.csv')
    args = parser.parse_args()
    return args


class GensimSummarizer(object):
    def __init__(self):
        self._summarize = summarize

    def summarize(self, input_lst, **kwargs):
        return self._summarize('\n'.join(input_lst), **kwargs)


def main():
    args = get_args()
    chats = load_kakao_csv(args.fname)
    
    summarizer = GensimSummarizer()
    print('Origin')
    print(chats)

    print('Summarize')
    print(summarizer.summarize(chats))

if __name__ == '__main__':

    main()
