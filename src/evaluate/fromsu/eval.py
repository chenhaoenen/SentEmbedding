# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/13 9:34 
# Description:
# --------------------------------------------
import scipy.stats
import numpy as np

def l2_normalize(vecs):
    """标准化
    from https://github.com/bojone/SimCSE/blob/main/utils.py
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    from https://github.com/bojone/SimCSE/blob/main/utils.py
    """
    return scipy.stats.spearmanr(x, y).correlation