# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/3 14:54 
# Description:  
# --------------------------------------------
import torch
import torch.nn as nn
from typing import Optional
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartEncoderLayer,
    PretrainedBartModel,
    BartLearnedPositionalEmbedding,
)

class ParaBART(PretrainedBartModel):
    def __init__(self, config):
        super(ParaBART, self).__init__(config)

    def forward(self):
        pass


class ParaBartEncoder(PretrainedBartModel):
    def __init__(self, config:BartConfig, embed_tokens:Optional[nn.Embedding]=None):
        super().__init__(config)
        self.config = config

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(self.max_source_positions, embed_dim, config.pad_token_id)
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # syntactic
        self.embed_synt = nn.Embedding(77, config.d_model, config.pad_token_id)
        self.embed_synt.weight.data.normal(mean=0.0, std=config.init_std)
        self.embed_synt.weight.data[config.pad_token_id].zero_()

        self.synt_layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(1)])
        self.synt_layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(self, token_input_ids=None, token_attention_mask=None, synt_input_ids=None, synt_attention_mask=None):

        if self.training:
            device = token_input_ids.device
            drop_mask = torch.bernoulli(self.config.word_dropout * torch.ones(token_input_ids.shape)).bool().to(device)
            token_input_ids = token_input_ids.masked_fill(drop_mask, 50264)

        # token embedding
        token_inputs_embeds = self.embed_tokens(token_input_ids)
        token_input_shape = token_input_ids.size()
        token_embed_pos = self.embed_positions(token_input_shape)
        token_hidden_states = token_inputs_embeds + token_embed_pos
        token_hidden_states = self.layernorm_embedding(token_hidden_states)
        token_hidden_states = nn.functional.dropout(token_hidden_states, p=self.dropout, training=self.training)





