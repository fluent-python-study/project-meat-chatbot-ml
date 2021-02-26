import csv
import json
import random

import numpy as np
import torch

def load_kakao_csv(fname:str):
    '''
    input : kakao message filename csv
    ex) '2020-03-20 00:01:19', 'kakao Eric', '왜 프로필 사진이 안바뀌지'

    output : str list
    ex) 'kakao Eric 왜 프로필 사진이 안바뀌지'
    '''
    chats = []
    with open(fname, newline='') as csvfile:
        chatreader = csv.reader(csvfile, delimiter=',')
        for row in chatreader:
            chat = ' '.join(row[1:]) # without date
            chats.append(chat)
    return chats[1:] # without header


def load_json(fname:str):
    with open(fname) as fp:
        obj = json.load(fp)
    return obj


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Extractor(object):
    def __init__(self, n_extracts):
        self.rouge = Rouge()
        self.n_extracts = n_extracts
    def extract(self, summary, original_articles):
        origins = [text for text in origial_articles if len(text) > 3]
        references = [summary for _ in origins]
        scores = self.rouge.get_scores(references, original_articles)
        results = [(score['rouge-1']['f'] + score['rouge-2']['f'] + score['rouge-3']['f'])/3 for score in scores]
        sorted_idxs = sorted(range(len(results)), key=lambda k:results[k], reverse=True)[:self.n_extracts]
        return [original_articles[idx] for idx in sorted_idxs]
