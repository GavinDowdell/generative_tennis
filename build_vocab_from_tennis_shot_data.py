# -*- coding: utf-8 -*-
"""
Created on Sun May 14 16:09:26 2023

@author: gavin
"""

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.functional as F

text = 'a224,b38,b3,b1,f2,f3,s3,f1,f3,b1,f1,f3*'

text1 = 'a224 b38 b3 b1 f2 f3 s3 f1 f3 b1 f1 f3*'

text.split(',')

tokenizer = get_tokenizer('basic_english')

tokenizer(text1)

myvocab = build_vocab_from_iterator(tokenizer(text1))

def yield_tokens(string_in):
    #yield tokenizer(string_in) 
    yield string_in.strip().split(' ')

def yield_tokens(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip().split(',')


vocab = build_vocab_from_iterator(yield_tokens(text1), 
        specials=["<unk>"],special_first=True) # ensure unknown is first

vocab.set_default_index(vocab["<unk>"])

vocab.get_itos()
vocab1.get_itos()

# works fine 
def yield_tokens(file_path):
    with open(file_path) as f:
        for line in f:
            #yield line.strip().split(',')
            yield tokenizer(line)
            
            
vocab1 = build_vocab_from_iterator(
    yield_tokens(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced_MLM.txt'), specials=["<unk>"])

vocab1()