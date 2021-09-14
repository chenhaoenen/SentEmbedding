# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/11 15:10 
# Description:  
# --------------------------------------------
import spacy
import collections
from spacy.symbols import ORTH
from typing import List, Dict, Union
# python3 -m spacy download en

class Tokenizer(object):
    def __init__(self, vocab:Dict[str, int], max_len:int=None, pad_token:str='</s>', unk_token:str='unk', do_lower_case=True):
        self.vocab = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.do_lower_case = do_lower_case
        assert self.pad_token in self.vocab and self.unk_token in self.vocab, f"padding token {pad_token} or unknown token {unk_token} is not in vocabulary"
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.max_len = max_len if max_len is not None else int(1e12)


    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def tokenize(self, text:str):

        raise NotImplementedError

    def encode_single_sent(self, text:str):
        tokens = self.tokenize(text)

        # padding and truncation
        if len(tokens) >= self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.pad_token] * (self.max_len - len(tokens))

        # lower and filter
        process_tokens = []
        for token in tokens:
            if self.do_lower_case:
                token = token.lower()
            token = token if token in self.vocab else self.unk_token
            process_tokens.append(token)

        return self.convert_tokens_to_ids(process_tokens)

    def encode(self, text:Union[str, List[str]]):
        if isinstance(text, str):
            return self.encode_single_sent(text)
        elif isinstance(text, list):
            return [self.encode_single_sent(sent) for sent in text]

    def __call__(self, text):
        return self.encode(text)



class SpacyTokenizer(Tokenizer):
    def __init__(self, vocab:Dict[str, int], max_len:int=64, pad_token:str='</s>', unk_token:str='unk', do_lower_case=True):
        super(SpacyTokenizer, self).__init__(vocab, max_len=max_len, pad_token=pad_token, unk_token=unk_token, do_lower_case=do_lower_case)

        self.vocab = vocab
        self.tokenizer = spacy.load('en_core_web_sm', parser=False, entity=False)
        self.add_special_token()

    def add_special_token(self):
        # for token in self.vocab:
        #     if token not in self.tokenizer.vocab:
        #         self.tokenizer.tokenizer.add_special_case(token, [{ORTH: token}])

        token = '</s>'
        self.tokenizer.tokenizer.add_special_case(token, [{ORTH: token}])


    def tokenize(self, text:str) -> List[str]:
        return [w.text for w in self.tokenizer(text)]

if __name__ == '__main__':
    text = '</s> I like _MATH_ even _MATH_ when _MATH_, except when _MATH_ is _MATH_! or _MATH_? or _MATH_: or _MATH_; or even _MATH_.. but not _MATH_. or _MATH_.'
    from util import load_word2vec

    voctors, w2i = load_word2vec('/code/NLPDataset/word2vec/GoogleNews-vectors-negative300.bin')
    tokenizer = SpacyTokenizer(w2i)

    a = tokenizer([text]*4)
    for w in a:
        print(w)


