# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:33:10 2023

@author: gavin
"""

import torch 
import numpy as np

x = torch.tensor([ 0,  1, 15, 15,  1,  0,  0,  0,  0,  0,  0])
y = torch.tensor([ 0,  1, 15, 15,  1,  0, -1, -1, -1, -1, -1])
y_orig = y.clone()

np.where(y == -1)[0][0]

# 0 for the masked target - the one we actually predict
tmp = np.random.binomial(list(np.ones(np.where(y == -1)[0][0])),0.8)
tmp

idx = np.where(tmp==1)[0].tolist()
idx
y[idx] = torch.tensor(-1)
y

block_size = 11
np.diag(np.ones(block_size))
# mask = torch.zeros(block_size,block_size) # use for no mask at all
mask = torch.diag(torch.ones(block_size))
mask
for idx,i in enumerate(y_orig):
    print(idx)
    print(i)
    if i < 0:
        mask[:,idx] = torch.ones(block_size)
        #mask[idx,:] = torch.ones(block_size)
        
mask

torch.tril(torch.ones(block_size,block_size))




x = torch.tensor([ 0,  1, 15, 15,  1,  0,  0,  0,  0,  0,  0])
y = torch.tensor([ 1, 15, 15,  1,  0, -1, -1, -1, -1, -1, -1])
tmp = np.random.binomial(list(np.ones(np.where(y == -1)[0][0])),0.8)
idx = np.where(tmp==1)[0].tolist()
y[idx] = torch.tensor(-1)
y






