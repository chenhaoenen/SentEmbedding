# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/13 22:29 
# Description: #use 中获取TF hub model 对应的sha1值
# --------------------------------------------
import sys

import hashlib

def get_sha1(url:str='https://tfhub.dev/google/universal-sentence-encoder-large/5'):
    return hashlib.sha1(url.encode("utf8")).hexdigest()

if __name__ == '__main__':
    url = sys.argv[1]
    sha1 = get_sha1(url)
    print(sha1)
    sys.exit(0)