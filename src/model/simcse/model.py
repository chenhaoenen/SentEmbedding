# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/11 16:11 
# Description:
# References:
# https://github.com/princeton-nlp/SimCSE
# https://spaces.ac.cn/archives/8348  苏神
# --------------------------------------------
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
import torch.distributed as dist
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

class Pooler(nn.Module):
    '''
    pooler to get sentence embedding

    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in {'cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'}


    def forward(self, outputs, attention_mask):
        '''
        :param outputs: dict
        :param attention_mask: [B, max_seq_length]
        :return:
        '''
        last_hidden = outputs.last_hidden_state #[B, seq_length, hidden_size]
        hidden_states = outputs.hidden_states #[#[B, seq_length, hidden_size], #[B, seq_length, hidden_size]...., #[B, seq_length, hidden_size]]

        if self.pooler_type in ['cls', 'cls_before_pooler']:
            return last_hidden[:,0]

        elif self.pooler_type == 'avg':
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            return (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        elif self.pooler_type == 'avg_top2':
            last_hidden = hidden_states[-1]
            second_last_hidden = hidden_states[-2]
            return (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        else:
            raise NotImplementedError

class MLPLayer(nn.Module):
    '''
    get sentence representations from Bert or RoBerta CLS
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activate = nn.Tanh()

    def forward(self, features):
        out = self.dense(features)
        out = self.activate(out)

        return out

class Similarity(nn.Module):
    '''
    dot product or cosine similarity
    '''
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SimCse(nn.Module):
    def __init__(self, pre_train_path, pooler_type, hard_negative_weight=None, temp=None):
        super(SimCse, self).__init__()
        config = AutoConfig.from_pretrained(pre_train_path)
        self.bert = AutoModel.from_pretrained(pre_train_path)
        self.pooler = Pooler(pooler_type)
        self.pooler_type = pooler_type
        if pooler_type == 'cls':
            self.mlp = MLPLayer(config)

        # loss
        self.hard_negative_weight = hard_negative_weight
        self.sim_fun = Similarity(temp)
        self.loss_fun = nn.CrossEntropyLoss()


    def forward(self, input_ids, token_type_ids, attention_mask, train=False):
        '''
        :param input_ids: [B, num_sent, sent_length]
        :param attention_mask: [B, num_sent, sent_length]
        :param token_type_ids: [B, num_sent, sent_length]
        :return:
        '''
        batch, num_sent, sent_length = input_ids.size()
        input_ids = input_ids.view((-1, sent_length)) #[B*num_sent, sent_length]
        attention_mask = attention_mask.view((-1, sent_length)) #[B*num_sent, sent_length]
        token_type_ids = token_type_ids.view((-1, sent_length)) #[B*num_sent, sent_length]

        # bert
        bert_outs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) #bert outputs dict

        # pooling
        pooler_out = self.pooler(bert_outs, attention_mask) #[B*num_sent, hidden_size]
        sent_embed_out = pooler_out.view((batch, num_sent, -1))  #[B, num_sent, hidden_size]

        if self.pooler_type == 'cls':
            sent_embed_out = self.mlp(sent_embed_out) #[B, num_sent, hidden_size]

        if train:
            loss = self.compute_loss(sent_embed_out)
            return loss
        else:
            return sent_embed_out.squeeze(1) # [B, hidden_size] when evaluate，num_sent equal 1


    def compute_loss(self, sent_embed_out):
        '''
        :param sent_embed_out: [B, num_sent, hidden_size]
        :return:
        '''
        device = sent_embed_out.device
        num_sent = sent_embed_out.size(1)

        sent1, sent2 = sent_embed_out[:,0, :], sent_embed_out[:, 1, :] # sent1:[B, hidden_size] sent2:[B, hidden_size]

        if num_sent == 3:
            sent3 = sent_embed_out[:, 2, :] #[B, hidden_size]

        if dist.is_initialized():

            # Gather all hard negative
            if num_sent == 3:
                sent3_list = [torch.zeros_like(sent3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=sent3_list, tensor=sent3.contiguous())
                sent3_list[dist.get_rank()] = sent3
                sent3 = torch.cat(sent3_list, dim=0) # [B*world_size, hidden_size]


            # Gather sent1 and sent2
            sent1_list = [torch.zeros_like(sent1) for _ in range(dist.get_world_size())]
            sent2_list = [torch.zeros_like(sent2) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=sent1_list, tensor=sent1.contiguous())
            dist.all_gather(tensor_list=sent2_list, tensor=sent2.contiguous())

            sent1_list[dist.get_rank()] = sent1
            sent2_list[dist.get_rank()] = sent2

            sent1 = torch.cat(sent1_list, dim=0) # [B*world_size, hidden_size]
            sent2 = torch.cat(sent2_list, dim=0) # [B*world_size, hidden_size]

        # calculate similarity
        cos_sim = self.sim_fun(sent1.unsqueeze(1), sent2.unsqueeze(0)) #[B*world_size, B*world_size]

        # Hard negative
        if num_sent == 3:
            sent1_sent3_cos_sim = self.sim_fun(sent1.unsqueeze(1), sent3.unsqueeze(0)) #[B*world_size, B*world_size]
            cos_sim = torch.cat([cos_sim, sent1_sent3_cos_sim], dim=1) # [B*world_size, 2*B*world_size]

        labels_sim = torch.arange(cos_sim.size(0)).long().to(device)

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            sent3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - sent1_sent3_cos_sim.size(-1)) + [0.0] * i + [sent3_weight] + [0.0] * (
                            sent1_sent3_cos_sim.size(-1) - i - 1) for i in range(sent1_sent3_cos_sim.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights
        loss = self.loss_fun(cos_sim, labels_sim)

        return loss

    def compute_flatnce_loss(self, sent_embed_out):
        '''
        https://spaces.ac.cn/archives/8586
        https://arxiv.org/abs/2107.01152
        '''

        '''
        :param sent_embed_out: [B, num_sent, hidden_size]
        :return:
        '''
        device = sent_embed_out.device
        num_sent = sent_embed_out.size(1)

        sent1, sent2 = sent_embed_out[:,0, :], sent_embed_out[:, 1, :] # sent1:[B, hidden_size] sent2:[B, hidden_size]

        if num_sent == 3:
            sent3 = sent_embed_out[:, 2, :] #[B, hidden_size]

        if dist.is_initialized():

            # Gather all hard negative
            if num_sent == 3:
                sent3_list = [torch.zeros_like(sent3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=sent3_list, tensor=sent3.contiguous())
                sent3_list[dist.get_rank()] = sent3
                sent3 = torch.cat(sent3_list, dim=0) # [B*world_size, hidden_size]


            # Gather sent1 and sent2
            sent1_list = [torch.zeros_like(sent1) for _ in range(dist.get_world_size())]
            sent2_list = [torch.zeros_like(sent2) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=sent1_list, tensor=sent1.contiguous())
            dist.all_gather(tensor_list=sent2_list, tensor=sent2.contiguous())

            sent1_list[dist.get_rank()] = sent1
            sent2_list[dist.get_rank()] = sent2

            sent1 = torch.cat(sent1_list, dim=0) # [B*world_size, hidden_size]
            sent2 = torch.cat(sent2_list, dim=0) # [B*world_size, hidden_size]

        # calculate similarity
        cos_sim = self.sim_fun(sent1.unsqueeze(1), sent2.unsqueeze(0)) #[B*world_size, B*world_size]

        # Hard negative
        if num_sent == 3:
            sent1_sent3_cos_sim = self.sim_fun(sent1.unsqueeze(1), sent3.unsqueeze(0)) #[B*world_size, B*world_size]
            cos_sim = torch.cat([cos_sim, sent1_sent3_cos_sim], dim=1) # [B*world_size, 2*B*world_size]

        labels_sim = torch.arange(cos_sim.size(0)).long().to(device)

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            sent3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - sent1_sent3_cos_sim.size(-1)) + [0.0] * i + [sent3_weight] + [0.0] * (
                            sent1_sent3_cos_sim.size(-1) - i - 1) for i in range(sent1_sent3_cos_sim.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights
        loss = self.loss_fun(cos_sim, labels_sim)



        return loss


