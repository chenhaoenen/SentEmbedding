# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/9 21:25 
# Description:  
# --------------------------------------------
def chunk(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l - n + 1):
        yield iterable[ndx:min(ndx + n, l)]