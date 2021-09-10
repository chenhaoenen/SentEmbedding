# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/8 10:02 
# Description:  
# --------------------------------------------
import torch
from torch import nn

class SkipThought(nn.Module):
    def __init__(self, config):
        super(SkipThought, self).__init__()

        self.config = config
        self.embed = nn.Embedding(config.vocab_size, embedding_dim=config.hidden_size)
        self.decoder1 = nn.GRU(input_size=2*config.hidden_size, hidden_size=config.hidden_size, batch_first=True)  # pre sent
        self.encoder2 = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)  # cur sent
        self.decoder3 = nn.GRU(input_size=2*config.hidden_size, hidden_size=config.hidden_size, batch_first=True)  # next sent

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.vocab_size)
        self.classifier3 = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, sent2, sent1=None, sent3=None):

        if sent1 is not None or sent3 is not None:
            loss = torch.tensor(0.0).to(sent2.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            s2_embed = self.embed(sent2)  # [B, seq_len, hidden_size]
            _, encoder_ht = self.encoder2(s2_embed)  # en2_out:[B, seq_len, hidden_size]; encoder_ht:[1, B, hidden_size]

            if sent1 is not None:
                s1_embed = self.embed(sent1)  # [B, seq_len, hidden_size]
                z1 = torch.transpose(encoder_ht.expand(s1_embed.size(1), -1, -1), 0, 1)  # [B, seq_len, hidden_size]
                de1_out, _ = self.decoder1(torch.cat([s1_embed, z1], dim=-1))  # [B, seq_len, hidden_size]
                de1_out = self.classifier1(de1_out)  # [B, seq_len, vocab_size]
                loss += loss_fct(de1_out[:, :-1, ].contiguous().view(-1, self.config.vocab_size), sent1[:, 1:, ].contiguous().view(-1))

            if sent3 is not None:
                s3_embed = self.embed(sent3)  # [B, seq_len, hidden_size]
                z3 = torch.transpose(encoder_ht.expand(s3_embed.size(1), -1, -1), 0, 1)  # [B, seq_len, hidden_size]
                de3_out, _ = self.decoder3(torch.cat([s3_embed, z3], dim=-1))  # [B, seq_len, hidden_size]
                de3_out = self.classifier3(de3_out)  # [B, seq_len, vocab_size]
                loss += loss_fct(de3_out[:, :-1, ].contiguous().view(-1, self.config.vocab_size), sent3[:, 1:, ].contiguous().view(-1))
            return loss
        else: # inference
            s2_embed = self.embed(sent2)  # [B, seq_len, hidden_size]
            _, encoder_ht = self.encoder2(s2_embed) # encoder_ht:[1, B, hidden_size]

            return encoder_ht.squeeze(0)
