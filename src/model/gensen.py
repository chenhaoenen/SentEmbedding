# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/14 15:31 
# Description: reference https://github.com/xingdi-eric-yuan/gensen/tree/py3pytorch.4
# --------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ConditionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(ConditionalGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_weights = nn.Linear(input_size, 3*hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 3*hidden_size)
        self.peep_weights = nn.Linear(hidden_size, 3*hidden_size)

    def forward(self, input, hidden, ctx):
        '''
        input: [B, trg_seq_len, emb_dim] emb_dim=input_size
        hidden: [B, src_hidden_dim]
        ctx: [B, src_hidden_dim]
        '''

        def recurrence(x, hidden, ctx):
            '''
            x: [B, input_size]
            hidden: [B, hidden_size]
            ctx: [B, hidden_size]
            '''
            input_gate = self.input_weights(x)
            hidden_gate = self.hidden_weights(hidden)
            peep_gate = self.peep_weights(ctx)
            i_r, i_i, i_n = input_gate.chunk(3, 1)
            h_r, h_i, h_n = hidden_gate.chunk(3, 1)
            p_r, p_i, p_n = peep_gate.chunk(3, 1)
            resetgate = torch.sigmoid(i_r + h_r + p_r)
            inputgate = torch.sigmoid(i_i + h_i + p_i)
            newgate = torch.tanh(i_n + resetgate * h_n + p_n)
            hy = newgate + inputgate * (hidden - newgate)
            return hy # [B, hidden_size]


        input = input.transpose(0, 1)
        output = []
        steps = input.size(0)
        for i in range(steps):
            hidden = recurrence(input[i], hidden, ctx)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
        output = torch.cat(output, dim=0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1) #[B, trg_seq_len, hidden_size]
        return output, hidden


class MutiTaskModel(nn.Module):
    '''
    A Multi task sequence to sequence model with GRU
    '''
    def __init__(self, src_vocab_size, src_embed_dim, src_pad_token_id, src_num_layers, src_hidden_dim,
                 trg_vocab_size, trg_embed_dim, trg_pad_token_id, trg_hidden_dim,
                 num_tasks, bidirectional=False, dropout=0.1,
                 ):
        super(MutiTaskModel, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.src_embed_dim = src_embed_dim
        self.src_pad_token_id = src_pad_token_id
        self.src_hidden_dim = src_hidden_dim
        self.src_num_layers = src_num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.src_embedding = nn.Embedding(src_vocab_size, src_embed_dim, src_pad_token_id)
        self.encoder = nn.GRU(input_size=src_embed_dim,
                              hidden_size=src_hidden_dim,
                              num_layers=src_num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              batch_first=True)
        self.enc_drp = nn.Dropout(dropout)

        self.trg_embedding = nn.ModuleList(
            [
                nn.Embedding(trg_vocab_size, trg_embed_dim, trg_pad_token_id)
                for _ in range(num_tasks)
            ]
        )

        # decoder
        self.decoders = nn.ModuleList(
            [
                ConditionalGRU(input_size=trg_embed_dim, hidden_size=trg_hidden_dim, dropout=dropout)
                for _ in range(num_tasks)
            ]
        )
        
        self.decoder2vocab = nn.ModuleList(
            [
                nn.Linear(trg_hidden_dim, trg_vocab_size)
                for _ in range(num_tasks)
            ]
        )

        # nli decoder
        self.nli_decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(4*src_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, batch, task_idx=None):
        if batch['type'] == 'nli':
            sent1, sent2 = batch['sent1'], batch['sent2'] # [B, max_seq_len]
            sent1_length, sent2_length = batch['sent1_length'],  batch['sent2_length'] # [B]
            sent1_emb = self.src_embedding(sent1) # [B, max_seq_len, emb_dim]
            sent1_emb = pack_padded_sequence(sent1_emb, sent1_length.cpu(), batch_first=True, enforce_sorted=False)
            _, sent1_h = self.encoder(sent1_emb)

            sent2_emb = self.src_embedding(sent2) # [B, max_seq_len, emb_dim]
            sent2_emb = pack_padded_sequence(sent2_emb, sent2_length.cpu(), batch_first=True, enforce_sorted=False)
            _, sent2_h = self.encoder(sent2_emb)

            if self.bidirectional:
                sent1_h = torch.cat([sent1_h[-1], sent1_h[-2]], dim=1)
                sent2_h = torch.cat([sent2_h[-1], sent2_h[-2]], dim=1)
            else:
                sent1_h = sent1_h[-1]
                sent2_h = sent2_h[-1]

            features = torch.cat([sent1_h, sent2_h, torch.abs(sent1_h - sent2_h), sent1_h * sent2_h], dim=-1)

            logit = self.nli_decoder(features)

            loss = F.cross_entropy(logit, batch['label'])

            return loss

        # sequence to sequence model
        else:
            src, trg_input_ids, trg_output_ids = batch['src'], batch['trg_input_ids'], batch['trg_output_ids']
            src_length = batch['src_length']  # [B]
            src_emb = self.src_embedding(src)

            trg_emb = self.trg_embedding[task_idx](trg_input_ids)

            src_emb = pack_padded_sequence(src_emb, src_length.cpu(), batch_first=True, enforce_sorted=False)
            _, src_ht = self.encoder(src_emb)

            if self.bidirectional:
                src_ht = torch.cat([src_ht[-1], src_ht[-2]], dim=1) #[B, 2*hidden_size]
            else:
                src_ht = src_ht[-1] #[B, hidden_size]

            src_ht = self.enc_drp(src_ht)

            trg_h, _ = self.decoders[task_idx](trg_emb, src_ht, src_ht)
            batch_size, max_len, hidden_size = trg_h.size() 
            trg_h_reshape = trg_h.contiguous().view(batch_size*max_len, -1)
            decoder_logit = self.decoder2vocab[task_idx](trg_h_reshape)

            loss = F.cross_entropy(decoder_logit, trg_output_ids.view(-1))

            return loss
