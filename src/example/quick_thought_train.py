# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/9 16:19
# Description:  
# --------------------------------------------
import os
import time
import torch
import random
import argparse
import numpy as np
from torch import optim
from typing import List
from functools import partial
from src.utils.util import chunk
from dataclasses import dataclass
from src.utils.timer import stats_time
from src.utils.util import load_word2vec
from src.evaluate.facebook import evaluate
from src.utils.tokenize import SpacyTokenizer
from src.model.quick_thought import QuickThought
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, set_seed, logging


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--epochs', default=10, type=int, help='The epoch of train')
    parser.add_argument('--pretrained_word2vec_path', default='/code/Demo/SimCSE/checkpoint/bert-base-uncased', type=str, help='The epoch of train')
    parser.add_argument('--input_path', default=10, type=str)
    parser.add_argument('--neg_num', default=5, type=int)
    parser.add_argument('--max_seq_length', default=32, type=int, help="The maximum length of squence")
    parser.add_argument('--batch_size', default=8, type=int, help="Total batch size for training.")
    parser.add_argument('--device', type=str, default='cpu', help='The devices id of gpu')
    parser.add_argument('--learning_rate', default=1.5e-5, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--log_freq', default=10, type=int, help='The freq of print log')

    args = parser.parse_args()

    return args


def setup_training(args):
    set_seed(args.seed)
    logging.set_verbosity_error()
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    args.device = device

    os.environ['TOKENIZERS_PARALLELISM'] = "true" #huggingface tokenizer ignore warning

    return args, device


def prepare_model_and_optimizer(args, device):

    @dataclass
    class Config():
        input_size: int
        hidden_size: int
        vectors: torch.FloatTensor

    vectors, vocab = load_word2vec(args.pretrained_word2vec_path)
    vectors = vectors[:100000,]
    vocab = {k:v for k,v in list(vocab.items())[:100000]}
    vocab_size, embed_dim = vectors.shape
    print(f'vocab size: {vocab_size}')
    tokenizer = SpacyTokenizer(vocab=vocab, unk_token='</s>')

    config = Config(vectors=torch.FloatTensor(vectors), input_size=embed_dim, hidden_size=768)
    model = QuickThought(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, optimizer, tokenizer


def bookCorpusDataLoader(args, tokenizer):

    class BookCorpusDataset(Dataset):
        def __init__(self, args):
            # load data
            examples = {}
            idx = 0
            with open(args.input_path, 'r') as reader:
                line = reader.readline()
                while line:
                    example = line.strip()
                    examples[idx] = example
                    line = reader.readline()
                    idx += 1
            # convert to feature
            features = {}
            for idx, example in examples.items():
                feat = tokenizer(example)
                features[idx] = torch.tensor(feat, dtype=torch.long)

            self.pairs = []
            for idxs in chunk(range(len(features)), 3):
                idx1, idx2, idx3 = idxs

                # positive
                self.pairs.append((features[idx2], features[idx1], 1.0))
                self.pairs.append((features[idx2], features[idx3], 1.0))

                # negative
                negs = []
                filte_idx = {idx1, idx2, idx3}
                while len(negs) < args.neg_num:
                    neg_idx = random.choice(range(len(features)))
                    if neg_idx not in filte_idx:
                        negs.append((features[idx2], features[neg_idx], 0.0))
                        filte_idx.add(neg_idx)
                self.pairs.extend(negs)

            np.random.shuffle(self.pairs)

            print('The num of example:', len(self.pairs))
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            return self.pairs[idx]


    train_data_loader = DataLoader(dataset=BookCorpusDataset(args),
                                   batch_size=args.batch_size,
                                   num_workers=4)
    return train_data_loader


def batch_func(args, model, tokenizer, device, batch:List[List[str]]):
    sentences = [' '.join(s) for s in batch]
    batch = tokenizer(sentences)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        outs = model(src=batch)
    return outs.cpu()


def trainer():
    args = parse_arguments()
    args, device = setup_training(args)

    model, optimizer, tokenizer = prepare_model_and_optimizer(args, device)
    train_data_loader = bookCorpusDataLoader(args, tokenizer)

    print(f"{'#' * 43} Args {'#' * 43}")
    for k in list(vars(args).keys()):
        print('{0}: {1}'.format(k, vars(args)[k]))

    total_step = args.epochs * len(train_data_loader)
    start = int(time.time())
    step = 0
    for epoch in range(args.epochs):
        #train
        print(f"{'#' * 41} Training {'#' * 41}")
        for i, batch in enumerate(train_data_loader):
            model.train()
            step += 1

            input = {}
            src, tgt, labels = batch
            input['src'] = src.to(device)
            input['tgt'] = tgt.to(device)
            input['labels'] = labels.to(device)

            loss = model(**input)
            loss.backward()
            if step % args.log_freq == 0:
                end = int(time.time())
                print(f"epoch:{epoch}, batch:{str(i+1)+'/'+str(len(train_data_loader))}, step:{str(step)+'/'+str(total_step)}, loss:{'{:.6f}'.format(loss)}, eta:{stats_time(start, end, step, total_step)}h, time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")
            optimizer.step()
            optimizer.zero_grad()

        print(f"{'#' * 40} Evaluating {'#' * 40}")
        res = evaluate(batch_func=partial(batch_func, args, model, tokenizer, device), eval_senteval_transfer=True)
        for k,v in res.items():
            print(f'{k}: {v}')


if __name__ == '__main__':
    trainer()
