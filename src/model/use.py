# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/13 22:20 
# Description:  
# --------------------------------------------
import spacy
import numpy as np

class Use(object):
    def __init__(self, model):
        self.nlp = spacy.load(model)
        print(f'Load model [{model}] successfully')

    def __call__(self, sent) -> np.ndarray:
        return self.nlp(sent).vector
