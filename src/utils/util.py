# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/9 21:25 
# Description:  
# --------------------------------------------
import torch
from gensim import models
from typing import Dict, Tuple

def chunk(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l - n + 1):
        yield iterable[ndx:min(ndx + n, l)]

def l2_norm(x:torch.Tensor, dim=-1):
    norm = ((x**2).sum(dim=dim, keepdim=True))**0.5
    return x/norm

def load_word2vec(path):
    model = models.KeyedVectors.load_word2vec_format(path, binary=True)
    voctors = model.vectors
    w2i = model.key_to_index
    return voctors, w2i

def trim_vocab(vocab: Dict[str, int], vocab_size: int = -1) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2id = {
        '<s>': 0,
        '<pad>': 1,
        '</s>': 2,
        '<unk>': 3
    }

    id2word = {
        0: '<s>',
        1: '<pad>',
        2: '</s>',
        3: '<unk>'
    }

    if '<s>' in vocab:
        del vocab['<s>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '</s>' in vocab:
        del vocab['</s>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    sorted_word2id = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    if vocab_size != -1:
        sorted_words = [x[0] for x in sorted_word2id[:vocab_size-4]]
    else:
        sorted_words = [x[0] for x in sorted_word2id]

    for idx, word in enumerate(sorted_words):
        word2id[word] = idx + 4
        id2word[idx+4] = word
    return word2id, id2word


if __name__ == '__main__':
    voctors, w2i = load_word2vec('/code/NLPDataset/word2vec/GoogleNews-vectors-negative300.bin')
    print(voctors.shape)
    a,b = voctors.shape
    print(a)
    print('hello world')
