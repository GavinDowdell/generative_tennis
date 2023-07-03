# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:46:22 2023

@author: gavin
"""
import torch
from torch import nn
X = torch.tensor([[ 0,  8,  5, 14, 15, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [ 0,  3, 15, 12,  5, 25,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
X
'''
batch size =2
tensor([[ 0,  8,  5, 14, 15, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  3, 15, 12,  5, 25,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
'''

X.size()
wte = nn.Embedding(26,10)
emb = wte(X)
emb.shape
rnn = nn.RNN(10,10,batch_first=True)
rnn(emb)
hidden, last = rnn(emb)
hidden.shape
last.shape
hidden[0,-1,:]
last[0,0,:]
lm_head = nn.Linear(10,2)
hidden.shape
lm_head(hidden)
lm_head(hidden).shape
lm_head(last)