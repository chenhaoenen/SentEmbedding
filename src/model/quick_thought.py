# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/10 17:08 
# Description:  
# --------------------------------------------
import torch
from torch import nn
from src.utils.util import l2_norm

class QuickThought(nn.Module):
    def __init__(self, config):
        super(QuickThought, self).__init__()

        self.embed = Embedding(config)
        self.encoderf = Encoder(config)
        self.encoderg = Encoder(config)

    def forward(self, src, tgt=None, labels=None):
        if tgt is not None and labels is not None:
            src_embed = self.embed(src)
            tgt_embed = self.embed(tgt)

            src_sent_embed = self.encoderf(src_embed)
            tgt_sent_embed = self.encoderg(tgt_embed)

            src_l2_embed = l2_norm(src_sent_embed) #[B, hidden_size]
            tgt_l2_embed = l2_norm(tgt_sent_embed) #[B, hidden_size]

            score = (src_l2_embed * tgt_l2_embed).sum(dim=-1) #[B]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)

            return loss
        else:
            src_embed = self.embed(src)
            src_sent_embed_f = self.encoderf(src_embed)
            src_sent_embed_g = self.encoderg(src_embed)
            src_l2_embed_f = l2_norm(src_sent_embed_f)  # [B, hidden_size]
            src_l2_embed_g = l2_norm(src_sent_embed_g)  # [B, hidden_size]
            return l2_norm(torch.cat([src_l2_embed_f, src_l2_embed_g], dim=-1))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.gru = nn.GRU(input_size=config.input_size, hidden_size=config.hidden_size, batch_first=True)

    def forward(self, x):
        '''
        x: [B, seq_len, input_size]
        '''
        _, ht_out = self.gru(x) #[1, B, hidden_size]

        return ht_out.squeeze(0) #[B, hidden_size]


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.vectors, padding_idx=0, freeze=False)

    def forward(self, input_ids):
        '''
        input_ids: [B, seq_len]
        '''
        return self.embedding(input_ids) #[B, seq_len, embed_dim]
