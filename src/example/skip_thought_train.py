# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/8 15:29 
# Description:  
# --------------------------------------------
import os
import time
import torch
import argparse
import numpy as np
from torch import optim
from typing import List
from functools import partial
from src.utils.util import chunk
from src.utils.timer import stats_time
from src.evaluate.facebook import evaluate
from src.model.skip_thought import SkipThought
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, set_seed, logging, AutoConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--epochs', default=10, type=int, help='The epoch of train')
    parser.add_argument('--pretrained_model_path', default='/code/Demo/SimCSE/checkpoint/bert-base-uncased', type=str, help='The epoch of train')
    parser.add_argument('--input_path', default=10, type=str)
    parser.add_argument('--max_seq_length', default=32, type=int, help="The maximum length of squence")
    parser.add_argument('--batch_size', default=8, type=int, help="Total batch size for training.")
    parser.add_argument('--device', type=str, default='cpu', help='The devices id of gpu')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--log_freq', default=10, type=int, help='The freq of print log')

    args = parser.parse_args()

    return args

def setup_training(args):
    set_seed(args.seed)
    logging.set_verbosity_error()
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    args.device = device

    os.environ['TOKENIZERS_PARALLELISM'] = "true" #huggingface tokenizer ignore warning

    return args, device

def prepare_model_and_optimizer(args, device):
    # use bert tokenizer and config file
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    bert_config = AutoConfig.from_pretrained(args.pretrained_model_path)
    model = SkipThought(bert_config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, optimizer, tokenizer

def bookCorpusDataLoader(args, tokenizer):

    # load data
    examples = []
    with open(args.input_path, 'r') as reader:
        line = reader.readline()
        while line:
            example = line.strip()
            examples.append(example)
            line = reader.readline()

    # convert to feature
    features = []
    for example in examples:
        feat = tokenizer(example, max_length=args.max_seq_length, truncation=True, padding='max_length').input_ids
        features.append(feat)

    triples = []
    for feat in chunk(features, 3):
        feat1, feat2, feat3 = feat
        triples.append((feat1, feat2, feat3))

    print('The num of example:', len(triples))

    #train dataloader
    np.random.shuffle(triples)

    sent1 = torch.tensor([f[0] for f in triples], dtype=torch.long)
    sent2 = torch.tensor([f[1] for f in triples], dtype=torch.long)
    sent3 = torch.tensor([f[2] for f in triples], dtype=torch.long)

    train_data_loader = DataLoader(dataset=TensorDataset(sent1, sent2, sent3),
                                   batch_size=args.batch_size,
                                   num_workers=4)
    return train_data_loader

def batch_func(args, model, tokenizer, device, batch:List[List[str]]):
    sentences = [' '.join(s) for s in batch]
    batch = tokenizer.batch_encode_plus(sentences, max_length=args.max_seq_length, truncation=True, padding='max_length', return_tensors='pt').input_ids
    model.eval()
    with torch.no_grad():
        outs = model(**{'sent2':batch.to(device)})
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
            sent1, sent2, sent3 = batch
            input['sent1'] = sent1.to(device)
            input['sent2'] = sent2.to(device)
            input['sent3'] = sent3.to(device)
            loss = model(**input)
            loss.backward()
            if step % args.log_freq == 0:
                end = int(time.time())
                print(f"epoch:{epoch}, batch:{str(i+1)+'/'+str(len(train_data_loader))}, step:{str(step)+'/'+str(total_step)}, loss:{'{:.6f}'.format(loss)}, eta:{stats_time(start, end, step, total_step)}h, time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")
            optimizer.step()
            optimizer.zero_grad()

        print(f"{'#' * 40} Evaluating {'#' * 40}")
        model.eval()
        res = evaluate(batch_func=partial(batch_func, args, model, tokenizer, device), eval_senteval_transfer=True)
        for k,v in res.items():
            print(f'{k}: {v}')


if __name__ == '__main__':
    trainer()
