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
    BartDecoderLayer,
    PretrainedBartModel,
    BartLearnedPositionalEmbedding,
)

class ParaBART(PretrainedBartModel):
    '''Reference https://github.com/uclanlp/ParaBART '''
    def __init__(self, config: BartConfig):
        super(ParaBART, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = ParaBartEncoder(config, self.shared)
        self.decoder = ParaBartDecoder(config, self.shared)


        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask=None,
                encoder_outputs=None,
                return_encoder_outputs=False):

        if attention_mask is None:
            attention_mask = input_ids == self.config.pad_token_id
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        if return_encoder_outputs:
            return encoder_outputs

        assert encoder_outputs is not None
        assert decoder_input_ids is not None

        decoder_input_ids = decoder_input_ids[:, :-1]

        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       attention_mask=decoder_attention_mask,
                                       )


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

        self.pooling = MeanPooling(config)

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

        for idx, encoder_layer in enumerate(self.layers):
            token_layer_outputs = encoder_layer(token_hidden_states, attention_mask=token_attention_mask)
            token_hidden_states = token_layer_outputs[0]

        # syntactic
        synt_inputs_embeds = self.embed_synt(synt_input_ids)
        synt_input_shape = synt_input_ids.size()
        synt_embed_pos = self.embed_positions(synt_input_shape)
        synt_hidden_states = synt_inputs_embeds + synt_embed_pos
        synt_hidden_states = self.synt_layernorm_embedding(synt_hidden_states)
        synt_hidden_states = nn.functional.dropout(synt_hidden_states, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.synt_layers):
            synt_layer_outputs = encoder_layer(synt_hidden_states, attention_mask=synt_attention_mask)
            synt_hidden_states = synt_layer_outputs[0]

        encoder_outputs = torch.cat([token_hidden_states, synt_hidden_states], dim=1)

        sent_embeds = self.pooling(token_hidden_states, token_input_ids)

        return encoder_outputs, sent_embeds


class ParaBartDecoder(PretrainedBartModel):
    def __init__(self, config:BartConfig, embed_tokens:Optional[nn.Embedding]=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        embed_dim = config.d_model

        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(self.max_source_positions, embed_dim, config.pad_token_id)

        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(1)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                decoder_cross_attn_head_mask=None):

        inputs_embeds = self.embed_tokens(input_ids)
        input_shape = input_ids.size()
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states,
                                          attention_mask=attention_mask,
                                          encoder_hidden_states=encoder_hidden_states,
                                          encoder_attention_mask=encoder_attention_mask,
                                          cross_attn_layer_head_mask=decoder_cross_attn_head_mask)
            hidden_states = layer_outputs[0]


        return hidden_states


class MeanPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, token_hidden_states, token_input_ids):
        mask = token_input_ids == self.config.pad_token_id
        mean_mask = mask.float() / mask.float().sum(1, keepdim=True)
        token_hidden_states = token_hidden_states * mean_mask.unsqueeze(2).sum(dim=1, keepdim=True)
        return token_hidden_states












