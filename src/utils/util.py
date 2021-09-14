# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/9 21:25 
# Description:  
# --------------------------------------------
import torch
from gensim import models

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

if __name__ == '__main__':
    voctors, w2i = load_word2vec('/code/NLPDataset/word2vec/GoogleNews-vectors-negative300.bin')
    print(voctors.shape)
    a,b = voctors.shape
    print(a)
    print('hello world')
