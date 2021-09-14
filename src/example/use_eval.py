# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/13 22:57 
# Description:  
# --------------------------------------------
import argparse
import numpy as np
from typing import List
from functools import partial
from src.model.use import Use
from src.evaluate.facebook import evaluate

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='en_use_lg')
    args = parser.parse_args()
    return args

def batch_func(model, batch:List[List[str]]):
    sentences = [' '.join(s) for s in batch]
    embeddings = [model(sent) for sent in sentences]
    return np.stack(embeddings, axis=0)

def evalutor():
    args = parse_arguments()
    model = Use(args.model_name)
    res = evaluate(batch_func=partial(batch_func, model), eval_senteval_transfer=True)
    for k, v in res.items():
        print(f'{k}: {v}')

if __name__ == '__main__':
    evalutor()