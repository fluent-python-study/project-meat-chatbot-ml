import os
import json

import requests
import py7zr

url = 'https://arxiv.org/src/1911.12237v2/anc/corpus.7z'
corpus_dir = './data/samsum_corpus'
zip_fn = './data/corpus.7z'


def main():
    if not os.path.isdir(corpus_dir):
        os.makedirs(corpus_dir)
    
    r = requests.get(url, allow_redirects=True)
    open(zip_fn, 'wb').write(r.content)
    
    archive = py7zr.SevenZipFile(zip_fn, mode='r')
    archive.extractall(path=corpus_dir)
    archive.close()
    
    os.remove(zip_fn)

if __name__ == '__main__':
    main()
