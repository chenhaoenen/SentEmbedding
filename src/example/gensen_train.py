# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/15 9:24 
# Description:  
# --------------------------------------------
import os
import time
import torch
import pickle
import random
import argparse
import itertools
import numpy as np
from torch import optim
from typing import Dict, List
from src.utils.util import trim_vocab
from src.utils.timer import stats_time
from dataclasses import dataclass, asdict
from src.model.gensen import MutiTaskModel
from transformers import set_seed, logging
from torch.utils.data import Dataset, IterableDataset, DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--vocab_dir', default='/code/SentEmbedding/data/gensen/vocab', type=str)
    parser.add_argument('--src_max_length', default=90, type=int)
    parser.add_argument('--trg_max_length', default=90, type=int)
    parser.add_argument('--lowercase', default=True, type=bool)

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float)
    parser.add_argument('--log_freq', default=100, type=int)

    args = parser.parse_args()

    return args

def setup_training(args):
    set_seed(args.seed)
    logging.set_verbosity_error()
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    args.device = device

    args.nmt = {
        'path':{
            'de-en': {
                'src_path': "data/gensen/corpora/nmt/training/train.nmt.de-en.en.tok",
                'trg_path': "data/gensen/corpora/nmt/training/train.nmt.de-en.de.tok",
                'taskname': "de-en"
            },
            'fr-en': {
                'src_path': "data/gensen/corpora/nmt/training/train.nmt.fr-en.en.tok",
                'trg_path': "data/gensen/corpora/nmt/training/train.nmt.fr-en.fr.tok",
                'taskname': "fr-en"
            }
        },

        'src_vocab_size': 80000,
        'src_embed_dim': 512,
        'src_pad_token_id': 1,
        'src_num_layers': 1,
        'src_hidden_dim': 2048,

        'trg_vocab_size': 30000,
        'trg_embed_dim': 512,
        'trg_pad_token_id': 1,
        'trg_hidden_dim': 2048

    }

    args.nli = {
        'train_path': "data/gensen/corpora/allnli.train.txt.clean.noblank",
         'dev_path': "data/gensen/corpora/snli_1.0_dev.txt.clean.noblank",
        'test_path': "data/gensen/corpora/snli_1.0_test.txt.clean.noblank",
    }

    return args, device

class BuildVocab():
    def __init__(self, args):
        self.vocab_dir = args.vocab_dir
        self.src_vocab_size = args.nmt['src_vocab_size']
        self.trg_vocab_size = args.nmt['trg_vocab_size']
        self.lowercase = args.lowercase

        self.tasknames = [p['taskname'] for p in args.nmt['path'].values()]
        self.fname_src = [p['src_path'] for p in args.nmt['path'].values()]
        self.fname_trg = [p['trg_path'] for p in args.nmt['path'].values()]
        self.f_src = [open(fname, 'r') for fname in self.fname_src]
        self.f_trg = [open(fname, 'r') for fname in self.fname_trg]

        self.src = {task: {'word2id': None, 'id2word': None} for task in self.tasknames}

        self.trg = {task: {'word2id': None, 'id2word': None} for task in self.tasknames}
        self.build_vocab()

    def build_vocab(self):
        if not os.path.exists(self.vocab_dir):
            raise ValueError(f'cant find vocab directory: {self.vocab_dir}')

        print('Building source vocabs ...')
        if os.path.exists(os.path.join(self.vocab_dir, 'src_vocab.pkl')):
            print('Found existing source vocab file, Reloading ...')
            vocab = pickle.load(open(os.path.join(self.vocab_dir, 'src_vocab.pkl'), 'rb'))
            word2id, id2word = vocab['word2id'], vocab['id2word']

        else:
            print('Could not find existing source vocab, Building ...')
            word2id, id2word = self.construct_vocab(itertools.chain.from_iterable(self.f_src), self.src_vocab_size, self.lowercase)
            pickle.dump({'word2id': word2id, 'id2word': id2word}, open(os.path.join(self.vocab_dir, 'src_vocab.pkl'), 'wb'))

        for task in self.src:  #即使多个src,也是同享同一套的词表
            self.src[task]['word2id'], self.src[task]['id2word'] = word2id, id2word

        print('Building target vocabs ...')
        if os.path.exists(os.path.join(self.vocab_dir, 'trg_vocab.pkl')):
            print('Found existing target vocab file Reloading ...')

            vocab = pickle.load(open(os.path.join(self.vocab_dir, 'trg_vocab.pkl'), 'rb'))
            for task, fname in zip(self.trg, self.f_trg):
                print(f'Reloading target vocab for {task}')
                word2id, id2word = vocab[task]['word2id'], vocab[task]['id2word']
                self.trg[task]['word2id'], self.trg[task]['id2word'] = word2id, id2word
        else:
            print('Could not find existing target vocab, Building ...')
            for task, fname in zip(self.trg, self.f_trg):
                print(f'Building vocab for {task}')
                word2id, id2word = self.construct_vocab(fname, self.trg_vocab_size, self.lowercase)
                self.trg[task]['word2id'], self.trg[task]['id2word'] = word2id, id2word

            pickle.dump(self.trg.copy(), open(os.path.join(self.vocab_dir, 'trg_vocab.pkl'), 'wb'))

        print('Building vocab Finished ...')

    def construct_vocab(self, sentences, vocab_size, lowercase=False, charlevel=False):
        vocab = {}
        for sentence in sentences:
            if isinstance(sentence, str):
                if lowercase:
                    sentence = sentence.lower()
                if not charlevel:
                    sentence = sentence.split()
            for word in sentence:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        print(f'Found {len(vocab)} words in dataset')
        word2id, id2word = trim_vocab(vocab, vocab_size)
        return word2id, id2word

def nmt_data_loader(args, taskname, src_vocab, trg_vocab):
    src_word2id = src_vocab[taskname]['word2id']
    trg_word2id = trg_vocab[taskname]['word2id']
    src_max_length = args.src_max_length
    trg_max_length = args.trg_max_length
    src_path = args.nmt['path'][taskname]['src_path']
    trg_path = args.nmt['path'][taskname]['trg_path']
    lowercase = args.lowercase

    class NmtDataset(IterableDataset):
        def __init__(self):
            self.src_path = src_path
            self.trg_path = trg_path

        def process_mapper(self, src:str, trg:str):
            src = src.strip().lower() if lowercase else src.strip()
            trg = trg.strip().lower() if lowercase else trg.strip()

            src_tokens = ['<s>'] + src.split()[:src_max_length-2] + ['</s>']
            trg_tokens = ['<s>'] + trg.split()[:trg_max_length-2] + ['</s>']

            src_length = len(src_tokens)
            trg_length = len(trg_tokens)

            src_tokens = src_tokens + ['<pad>'] * (src_max_length-src_length)
            trg_tokens = trg_tokens + ['<pad>'] * (trg_max_length-trg_length)

            src_ids = torch.tensor([src_word2id[token] if token in src_word2id else src_word2id['<unk>'] for token in src_tokens], dtype=torch.long)
            trg_input_ids = torch.tensor([trg_word2id[token] if token in trg_word2id else trg_word2id['<unk>'] for token in trg_tokens[:-1]], dtype=torch.long)
            trg_output_ids = torch.tensor([trg_word2id[token] if token in trg_word2id else trg_word2id['<unk>'] for token in trg_tokens[1:]], dtype=torch.long)

            src_length = torch.tensor(src_length, dtype=torch.int64)

            return src_ids, trg_input_ids, trg_output_ids, src_length

        def __iter__(self):
            src = open(self.src_path, 'r')
            trg = open(self.trg_path, 'r')
            return map(self.process_mapper, src, trg)

    data_loader = DataLoader(dataset=NmtDataset(), batch_size=args.batch_size)

    return data_loader

def nli_data_loader(args, vocab):
    word2id = vocab['de-en']['word2id']
    train_path = args.nli['train_path']
    dev_path = args.nli['dev_path']
    test_path = args.nli['test_path']
    max_length = args.src_max_length
    lowercase = args.lowercase

    text2label = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    class NliDataset(Dataset):
        def __init__(self):
            self.data = [open(train_path, 'r'), open(dev_path, 'r'), open(test_path, 'r')]
            self.data = [d.strip().split('\t') for d in itertools.chain.from_iterable(self.data)]
            random.shuffle(self.data)
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sent1, sent2, label = self.data[idx]
            return sent1, sent2, label

    def collate_fn(batch):
        sent1s, sent2s, labels = zip(*batch)

        sent1s = [sent1.lower() if lowercase else sent1 for sent1 in sent1s]
        sent2s = [sent2.lower() if lowercase else sent2 for sent2 in sent2s]

        sent1s = [['<s>'] + sent1.split()[:max_length - 2] + ['</s>'] for sent1 in sent1s]
        sent2s = [['<s>'] + sent2.split()[:max_length - 2] + ['</s>'] for sent2 in sent2s]

        sent1s_length = [len(sent1) for sent1 in sent1s]
        sent2s_length = [len(sent2) for sent2 in sent2s]

        sent1s_tokens = [sent1 + ['<pad>'] * (max(sent1s_length) - len(sent1)) for sent1 in sent1s]
        sent2s_tokens = [sent2 + ['<pad>'] * (max(sent2s_length) - len(sent2)) for sent2 in sent2s]

        sent1s_ids = torch.tensor([[word2id[token] if token in word2id else word2id['<unk>'] for token in sent1] for sent1 in sent1s_tokens], dtype=torch.long)
        sent2s_ids = torch.tensor([[word2id[token] if token in word2id else word2id['<unk>'] for token in sent2] for sent2 in sent2s_tokens], dtype=torch.long)

        sent1s_length = torch.tensor(sent1s_length, dtype=torch.int64)
        sent2s_length = torch.tensor(sent2s_length, dtype=torch.int64)

        labels = torch.tensor([text2label[l] for l in labels], dtype=torch.long)

        return sent1s_ids, sent1s_length, sent2s_ids, sent2s_length, labels


    data_loader = DataLoader(dataset=NliDataset(), batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    return data_loader

def prepare_model_and_optimizer(args, device):

    @dataclass
    class Config():
        src_vocab_size: int = args.nmt['src_vocab_size']
        src_embed_dim: int = args.nmt['src_embed_dim']
        src_pad_token_id: int = args.nmt['src_pad_token_id']
        src_num_layers: int = args.nmt['src_num_layers']
        src_hidden_dim: int = args.nmt['src_hidden_dim']

        trg_vocab_size: int = args.nmt['trg_vocab_size']
        trg_embed_dim: int = args.nmt['trg_embed_dim']
        trg_pad_token_id: int = args.nmt['trg_pad_token_id']
        trg_hidden_dim: int = args.nmt['trg_hidden_dim']

        num_tasks: int = 3

    config = asdict(Config())
    model = MutiTaskModel(**config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, optimizer

def trainer():

    args = parse_arguments()
    args, device = setup_training(args)
    vocab = BuildVocab(args)
    de_en_dataloader = iter(nmt_data_loader(args, 'de-en', vocab.src, vocab.trg))

    fr_en_dataloader = iter(nmt_data_loader(args, 'fr-en', vocab.src, vocab.trg))

    nli_dataloader = iter(nli_data_loader(args, vocab.src))

    model, optimizer = prepare_model_and_optimizer(args, device)

    nli_ctr = 0
    step = 0
    start = int(time.time())
    task_idxs_dic = {0:'nli', 1:'de-en', 2:'fr-en'}
    nli_losses = []; nmt_losses = []
    total_step = 1000000
    while True:
        step += 1

        if step % 10 == 0:
            nli_batch = next(nli_dataloader, None)
            if nli_batch is None:
                nli_dataloader = iter(nli_data_loader(args, vocab.src))
                nli_batch = next(nli_dataloader)

            sent1s_ids, sent1s_length, sent2s_ids, sent2s_length, labels = [w.to(device) for w in nli_batch]
            batch = {}
            batch['type'] = 'nli'
            batch['sent1'] = sent1s_ids
            batch['sent2'] = sent2s_ids
            batch['sent1_length'] = sent1s_length
            batch['sent2_length'] = sent2s_length
            batch['label'] = labels
            task_idx = 0
            loss = model(batch=batch, task_idx=task_idx)
            nli_losses.append(loss.item())
            nli_ctr += 1

        else:
            task_idx = np.random.randint(low=1, high=len(task_idxs_dic)-1)

            if task_idxs_dic[task_idx] == 'de-en':
                nmt_batch = next(de_en_dataloader, None)
                if nmt_batch is None:
                    de_en_dataloader = iter(nmt_data_loader(args, 'de-en', vocab.src, vocab.trg))
                    nmt_batch = next(de_en_dataloader)

            elif task_idxs_dic[task_idx] == 'fr-en':
                nmt_batch = next(fr_en_dataloader, None)
                if nmt_batch is None:
                    fr_en_dataloader = iter(nmt_data_loader(args, 'fr-en', vocab.src, vocab.trg))
                    nmt_batch = next(fr_en_dataloader)
            else:
                raise ValueError(f'Could found task_name:{task_idxs_dic[task_idx]}')

            src_ids, trg_input_ids, trg_output_ids, src_length = [w.to(device) for w in nmt_batch]

            batch = {}
            batch['type'] = task_idxs_dic[task_idx]
            batch['src'] = src_ids
            batch['trg_input_ids'] = trg_input_ids
            batch['src_length'] = src_length
            batch['trg_output_ids'] = trg_output_ids
            loss = model(batch, task_idx)
            nmt_losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        if step % args.log_freq == 0:
            end = int(time.time())
            print(f"step:{str(step) + '/' + str(total_step)} | nli step: {nli_ctr} | nli loss:{'{:.6f}'.format(np.mean(nli_losses))} | nmt loss:{'{:.6f}'.format(np.mean(nmt_losses))} | eta:{stats_time(start, end, step, total_step)}h | time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")

            nli_losses = []; nmt_losses = []

if __name__ == '__main__':
    trainer()
