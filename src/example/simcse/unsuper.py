# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/12 9:32 
# Description:  
# --------------------------------------------
import os
import time
import glob
import torch
import argparse
import numpy as np
from torch import optim
from src.utils.timer import stats_time
from src.model.simcse.model import SimCse
from transformers import AutoTokenizer, set_seed
from torch.utils.data import TensorDataset, DataLoader
from src.evaluate.fromsu.eval import l2_normalize, compute_corrcoef

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--epochs', default=10, type=int, help='The epoch of train')
    parser.add_argument('--pretrained_model_path', default='/code/Demo/SimCSE/checkpoint/bert-base-uncased', type=str, help='The epoch of train')
    parser.add_argument('--pooler_type', default='cls', type=str, help='bert embedding pooler type')
    parser.add_argument('--temp', default=0.05, type=float, help='Temperature for softmax')
    parser.add_argument('--hard_negative_weight', default=0, type=float, help='The **logit** of weight for hard negatives (only effective if hard negatives are used')
    parser.add_argument('--task_dir', required=True, type=str, help="The directory of train data")
    parser.add_argument('--task_name', required=True, type=str, help="The task name of train data")
    parser.add_argument('--max_seq_length', default=128, type=int, help="The maximum length of squence")
    parser.add_argument('--batch_size', default=8, type=int, help="Total batch size for training.")
    parser.add_argument('--device', type=str, default='cpu', help='The devices id of gpu')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--log_freq', default=10, type=int, help='The freq of print log')

    args = parser.parse_args()

    return args

def setup_training(args):
    set_seed(args.seed)
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    assert os.path.isdir(args.pretrained_model_path), f"pre-training model path:{args.pre_trained_model_path} is not exists"

    args.input_dir = os.path.join(args.task_dir, args.task_name)
    assert os.path.isdir(args.input_dir), f"input dir:{args.input_dir} is not exists"

    os.environ['TOKENIZERS_PARALLELISM'] = "true" #huggingface tokenizer ignore warning

    return args, device

def prepare_model_and_optimizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    model = SimCse(pre_train_path=args.pretrained_model_path, pooler_type=args.pooler_type, hard_negative_weight=args.hard_negative_weight, temp=args.temp).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, optimizer, tokenizer

def sentevalCNDataLoader(args, tokenizer):

    # load data
    examples = []
    for file in glob.glob(args.input_dir+'/*.data'):
        with open(file, 'r') as reader:
            line = reader.readline()
            while line:
                content = line.strip().split('\t')
                if len(content) == 3:
                    senta, sentb, label = content[0], content[1], float(content[2])
                    examples.append((senta, sentb, label))
                line = reader.readline()

    # convert to feature
    train_feats = []
    senta_feats = []
    sentb_feats = []
    labels = []
    for example in examples:
        senta, sentb, label = example
        feata = tokenizer(senta, max_length=args.max_seq_length, truncation=True, padding='max_length')
        featb = tokenizer(sentb, max_length=args.max_seq_length, truncation=True, padding='max_length')
        senta_feats.append(feata)
        sentb_feats.append(featb)
        train_feats.append(feata)
        train_feats.append(featb)
        labels.append(label)

    #train dataloader
    np.random.shuffle(train_feats)
    train_feats = train_feats[:10000]
    input_ids = torch.tensor([f.input_ids for f in train_feats], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in train_feats], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in train_feats], dtype=torch.long)

    train_datasets = TensorDataset(input_ids, token_type_ids, attention_mask)
    def collate_fn(batch):
        input_ids, token_type_ids, attention_mask = zip(*batch)

        input_ids = torch.stack(input_ids).unsqueeze(1).repeat(1,2,1)  #[B, num_sent=2, max_seq_length]
        token_type_ids = torch.stack(token_type_ids).unsqueeze(1).repeat(1,2,1)
        attention_mask = torch.stack(attention_mask).unsqueeze(1).repeat(1,2,1)

        return input_ids, token_type_ids, attention_mask

    train_data_loader = DataLoader(dataset=train_datasets,
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn,
                                   num_workers=4)

    # eval datasets
    def eval_data_loader(sent_feats):

        sent_input_ids = torch.tensor([f.input_ids for f in sent_feats], dtype=torch.long).unsqueeze(1)
        sent_token_type_ids = torch.tensor([f.token_type_ids for f in sent_feats], dtype=torch.long).unsqueeze(1)
        sent_attention_mask = torch.tensor([f.attention_mask for f in sent_feats], dtype=torch.long).unsqueeze(1)

        sent_datasets = TensorDataset(sent_input_ids, sent_token_type_ids, sent_attention_mask)

        data_loader = DataLoader(dataset=sent_datasets,
                                 batch_size=args.batch_size*2,
                                 num_workers=4)
        return data_loader


    return train_data_loader, (eval_data_loader(senta_feats), eval_data_loader(sentb_feats), labels)

def eval_corrcoef(senta_embeds, sentb_embeds, labels):
    senta_vecs = l2_normalize(senta_embeds)
    sentb_vecs = l2_normalize(sentb_embeds)

    sims = (senta_vecs * sentb_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)

    return corrcoef

def trainer():
    args = parse_arguments()
    args, device = setup_training(args)

    model, optimizer, tokenizer = prepare_model_and_optimizer(args, device)
    train_data_loader, (senta_eval_data_loader, sentb_eval_data_loader, labels) = sentevalCNDataLoader(args, tokenizer)

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
            batch = [w.to(device) for w in batch]
            input_ids, token_type_ids, attention_mask = batch
            loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, train=True)

            loss.backward()
            if step % args.log_freq == 0:
                end = int(time.time())
                print(f"epoch:{epoch}, batch:{str(i+1)+'/'+str(len(train_data_loader))}, step:{str(step)+'/'+str(total_step)}, loss:{'{:.6f}'.format(loss)}, eta:{stats_time(start, end, step, total_step)}h, time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")
            optimizer.step()
            optimizer.zero_grad()

        print(f"{'#' * 40} Evaluating {'#' * 40}")
        model.eval()

        #senta and sentb embedding
        senta_embeds = []
        sentb_embeds = []
        with torch.no_grad():
            for batch in senta_eval_data_loader:
                batch = [w.to(device) for w in batch]
                input_ids, token_type_ids, attention_mask = batch
                embed = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                senta_embeds.append(embed.cpu())
            for batch in sentb_eval_data_loader:
                batch = [w.to(device) for w in batch]
                input_ids, token_type_ids, attention_mask = batch
                embed = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                sentb_embeds.append(embed.cpu())
        senta_embeds = torch.cat(senta_embeds, dim=0).numpy()
        sentb_embeds = torch.cat(sentb_embeds, dim=0).numpy()

        print(f'task_name:{args.task_name}, corrcoef:{eval_corrcoef(senta_embeds, sentb_embeds, labels)}')



if __name__ == '__main__':
    trainer()
