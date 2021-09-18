# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/9 09:03
# Description:  
# --------------------------------------------
import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartConfig,
    BartModel,
    shift_tokens_right
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.file_utils import (
    add_start_docstrings_to_model_forward
)


class BartSkipThought(BartPretrainedModel):
    base_model_prefix = 'model'
    def __init__(self, config:BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        sent2,
        sent1=None,
        sent3=None,
    ):

        if sent1 is not None or sent3 is not None:
            loss = torch.tensor(0.0).to(self.model.device)

            if sent1 is not None:
                loss += self.decoder(src_input_ids=sent2['input_ids'],
                                     src_attention_mask=sent2['attention_mask'],
                                     tgt_attention_mask=sent1['attention_mask'],
                                     labels=sent1['input_ids'])
            if sent3 is not None:
                loss += self.decoder(src_input_ids=sent2['input_ids'],
                                     src_attention_mask=sent2['attention_mask'],
                                     tgt_attention_mask=sent3['attention_mask'],
                                     labels=sent3['input_ids'])

            return loss

        else: #inference
            encoder_hidden_states = self.model.encoder(input_ids=sent2['input_ids'], attention_mask=sent2['attention_mask'])[0] #[B, seq_length, hidden_size]
            return (encoder_hidden_states * sent2['attention_mask'].unsqueeze(-1)).sum(1) / sent2['attention_mask'].sum(-1).unsqueeze(-1) #[B, hidden_size]

    def decoder(self, src_input_ids, src_attention_mask, tgt_attention_mask, labels):

        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        outputs = self.model(
            src_input_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=tgt_attention_mask,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        return loss


if __name__ == '__main__':

    device = torch.device('cpu')
    config = transformers.AutoConfig.from_pretrained('/code/pre_trained_model/model/bart-base')
    model = BartSkipThought(config).from_pretrained('/code/pre_trained_model/model/bart-base').to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('/code/pre_trained_model/model/bart-base')
    reader = open('/code/SentEmbedding/data/book_corpus_small/input.txt', 'r')
    examples = [e.strip() for e in reader.readlines()[:4]]

    sent1 = examples[:-1]; sent2 = examples[1:]

    feat1 = tokenizer(sent1[0], max_length=64, truncation=True, padding='max_length', return_tensors='pt').to(device)
    feat2 = tokenizer(sent2[0], max_length=64, truncation=True, padding='max_length', return_tensors='pt').to(device)

    print(feat1)
    # out = model(input_ids=feat1.input_ids,
    #             attention_mask=feat1.attention_mask,
    #             # decoder_input_ids=feat2.input_ids,
    #             decoder_attention_mask=feat2.attention_mask[:,1:,],
    #             labels = feat2.input_ids[:,1:]
    #             )
    #
    # print(out)
    # print(out.size())
    # # print(type(out))
    # print(len(out))
