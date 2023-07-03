"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

'''
to run in spyder or ipython

python makemore_spyder.py -i names.txt -o names --type mlp --max-steps 10000
python makemore_spyder.py -i names.txt -o names --type mlp --sample-only


run makemore_spyder.py -i names_short.txt -o names --type mlp --max-steps 10000
run makemore_spyder.py -i names_short.txt -o names --type mlp --sample-only

Note: For sample-only it is loading a pretrained model from the output directory. 
However the model that you call here will be instantiated hence it must be the 
same model structure - in other words you cannot load mlp weights into
a transformer 


can always load the weights in and have a look at the dict to check the model type
tmp = torch.load('C:\gavin\software\python\pytorch\karpathy\makemore\makemore\out\model.pt')
tmp
tmp.keys()

'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

#%%

#global train_dataset

#%%

os.chdir('C:\gavin\software\python\pytorch\karpathy\makemore\makemore')
print(os.getcwd())

#%% -----------------------------------------------------------------------------

# dataclass is a new python feature
# basically a class that just stores data
# it is just a normal class without inheritance

@dataclass
class ModelConfig:
    # Bear in mind these values of block_size and vocab_size are None here
    # but they are always initialised by the particular model
    # maximum length of the sequence or number of char in the words
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

tmp = ModelConfig()
tmp.n_head
tmp.block_size
tmp = ModelConfig(n_head = 2)
tmp.n_head
tmp.block_size

#%% -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

#%%
# Causal and not masked hence predicting the character at the end of the sequence
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # generate the 3 inputs query, key,value - but all different?
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        # notice how the output projection is of dimension n_embd
        # the transformer contextualises the embeddings but same dim
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        # takes the original input embedding vector and makes 3 versions for the 
        # query, key and value
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        # a different version of each is now generated for each head
        # each head must see a different version otherwise it won't learn anything different 
        # for each head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

#%%

# does self attention plus other things to put in a single encoder block
# just the layer norms and the additional dense projections
# this actually completes the encoder layer
# all the transformer class does after this is to combine a number of these 
# layers and decode over the output
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

#%%

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # setup a dict - can access individual or all or once
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # just putting the enocder blocks together on a Dict
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # the output of the transformer is dim n_embd which then
        # gets decoded to the output over the vocab
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        # start by simply embedding the integer tokens
        tok_emb = self.transformer.wte(idx) 
        # position embeddings of shape (1, t, n_embd)
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

#%%

# NOTE IF YOU USE THIS NEED TO CHANGE THE CALL IN GENERATE AT LINE 1007 OR NEARBY DUE TO THE MASK
class MyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # assign for later
        self.enc_layer = torch.nn.TransformerEncoderLayer(d_model=config.n_embd,nhead=config.n_head, dim_feedforward=config.n_embd, batch_first=True)  # d_model divisible by nhead
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            trans_enc = torch.nn.TransformerEncoder(self.enc_layer,num_layers=config.n_layer)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def get_block_size(self):
        return self.block_size
    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        hiddens = []
        for i in range(t):

            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            # forward the GPT model itself
            # token embeddings of shape (b, t, n_embd)
            # start by simply embedding the integer tokens
            tok_emb = self.transformer.wte(idx) 
            # position embeddings of shape (1, t, n_embd)
            pos_emb = self.transformer.wpe(pos) 
            x = tok_emb + pos_emb
            #for block in self.transformer.h:
            #   x = block(x)
            #x = self.transformer.ln_f(x)
            
            # at this point need to generate and pass a mask
            if isinference:
                x = self.transformer.trans_enc(src = x)
            else:
                #mask = torch.tril(torch.ones(config.block_size, config.block_size)) == 0
                mask = nn.Transformer.generate_square_subsequent_mask(config.block_size)
                x = self.transformer.trans_enc(src = x,mask = mask)
            #print(x.shape)
            #x = x.mean(axis=1)
            #print(x.shape)
            #input()
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
        
        

#%% -----------------------------------------------------------------------------
# Bag of Words (BoW) language model
# Causal meaning the previous words predicting the next word not the masked model
class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, config):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        # do the weighted average of all preceeding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

#%%

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(config)
        # MLP assembler
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x
#%%

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # context block
        self.context_block = BoWBlock(config)
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the token and position embedding layers
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # add and run through the decoder MLP
        x = tok_emb + pos_emb
        # run the bag of words context module
        x = self.context_block(x)
        # decode to next token probability
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

#%% -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
    # normally calculate the hidden layer vecotrs twice
    # then add them (last hidden plus estimated hidden from current token  
    # that is the standard way 
    # however this is equivalent
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

#%%

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

#%%

class RNN(nn.Module):
    # instantiate with config
    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # get the starting token embedding and starting hidden state
        # the starting hidden state
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) 
        # token embeddings table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) 
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            # emb already calculated per idx
            xt = emb[:, i, :] # (b, n_embd)
            # now get the new hidden
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            # capture all of the hidden states as it is from these 
            # that we make the predictions
            hiddens.append(ht)
            # only use the current token to predict the bext one
            #hiddens.append(xt)

        # decode the outputs
        # (b, t, n_embd2) - batch, timesteps,num units in hidden layer 
        # - ultimate extract the final hidden state at the final timestep for each batch
        # maybe not in this case as it is predicting at each timestep
        
        # essentially this turns a list into a tensor ready for a linear layer
        hidden = torch.stack(hiddens, 1) 
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # the cross_entropy loss activates the logit
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # looks like here logits.view(-1, logits.size(-1)) will extract the last hidden state only

        return logits, loss

#%%


class MLPMine1(nn.Module):
    # instantiate with config
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # get the starting token embedding and starting hidden state
        # the starting hidden state
        # self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) 
        # token embeddings table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) 
        #if cell_type == 'rnn':
        #    self.cell = RNNCell(config)
        #elif cell_type == 'gru':
        #    self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)
        # want to keep the option of having a hidden layer
        self.mlp = nn.Sequential(
            # so if the block size is say the max num of characters in any
            # word and the embedding size is 64 then you need
            # block_size * emb dim in the first layer as these embeddings 
            # get concatenated together
            nn.Linear(config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size) # decode to the output here
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        #hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            # emb already calculated per idx
            xt = emb[:, i, :] # (b, n_embd)
            # now get the new hidden
            #ht = self.cell(xt, hprev) # (b, n_embd2)
            #hprev = ht
            #hiddens.append(ht)
            # only use the current token to predict the bext one
            hiddens.append(xt)

        # decode the outputs over the prob distribution of the targets
        # essentially this turns a list into a tensor ready for a linear layer
        hidden = torch.stack(hiddens, 1) 
        #logits = self.lm_head(hidden)
        logits = self.mlp(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # the cross_entropy loss activates the logit
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # looks like here logits.view(-1, logits.size(-1)) will extract the last hidden state only

        return logits, loss


#%%

class MLPMine2(nn.Module):
    # instantiate with config
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # get the starting token embedding and starting hidden state
        # the starting hidden state
        #self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) 
        # token embeddings table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) 
        #if cell_type == 'rnn':
        #    self.cell = RNNCell(config)
        #elif cell_type == 'gru':
        #    self.cell = GRUCell(config)
        self.lm_head = nn.Linear(2*config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        #hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            # emb already calculated per idx
            #xt = emb[:, i, :] # (b, n_embd)
            # now get the new hidden
            #ht = self.cell(xt, hprev) # (b, n_embd2)
            #hprev = ht
            #hiddens.append(ht)
            # only use the current token to predict the bext one
            if i > 0:
                xt = torch.cat([emb[:, i-1, :],emb[:, i, :]],dim=1)
            else:
                xt = torch.cat([emb[:, i, :],emb[:, i, :]],dim=1)
                
            hiddens.append(xt)

        # decode the outputs over the prob distribution of the targets
        # essentially this turns a list into a tensor ready for a linear layer
        hidden = torch.stack(hiddens, 1) 
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # the cross_entropy loss activates the logit
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # looks like here logits.view(-1, logits.size(-1)) will extract the last hidden state only

        return logits, loss
#%%

class MLPMine3(nn.Module):
    # instantiate with config
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # get the starting token embedding and starting hidden state
        # the starting hidden state
        #self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) 
        # token embeddings table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) 
        #if cell_type == 'rnn':
        #    self.cell = RNNCell(config)
        #elif cell_type == 'gru':
        #    self.cell = GRUCell(config)
        self.lm_head = nn.Linear(3*config.n_embd, self.vocab_size)
        self.mlp = nn.Sequential(
            # so if the block size is say the max num of characters in any
            # word and the embedding size is 64 then you need
            # block_size * emb dim in the first layer as these embeddings 
            # get concatenated together
            nn.Linear(3 * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size) # decode to the output here
        )
        # save the embedding that is first initialised
        #print("\nPrint out initial emb\n")
        #emb_save_init = self.wte.weight.detach().numpy()
        #np.savetxt('emb_save_init.txt',emb_save_init, delimiter=',')
        

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        #hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            # emb already calculated per idx
            #xt = emb[:, i, :] # (b, n_embd)
            # now get the new hidden
            #ht = self.cell(xt, hprev) # (b, n_embd2)
            #hprev = ht
            #hiddens.append(ht)
            # only use the current token to predict the bext one
            if i == 0:
                xt = torch.cat([emb[:, i, :],emb[:, i, :],emb[:, i, :]],dim=1)
            elif i == 1:
                xt = torch.cat([emb[:, i-1, :],emb[:, i-1, :],emb[:, i, :]],dim=1)
            else:
                xt = torch.cat([emb[:, i-2, :],emb[:, i-1, :],emb[:, i, :]],dim=1)
                
            hiddens.append(xt)

        # decode the outputs over the prob distribution of the targets
        # essentially this turns a list into a tensor ready for a linear layer
        hidden = torch.stack(hiddens, 1) 
        #logits = self.lm_head(hidden)
        logits = self.mlp(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # the cross_entropy loss activates the logit
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # looks like here logits.view(-1, logits.size(-1)) will extract the last hidden state only

        return logits, loss

#%%

class CBOW(nn.Module):
    # instantiate with config
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # get the starting token embedding and starting hidden state
        # the starting hidden state
        #self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) 
        # token embeddings table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) 
        #if cell_type == 'rnn':
        #    self.cell = RNNCell(config)
        #elif cell_type == 'gru':
        #    self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)
        self.mlp = nn.Sequential(
            # so if the block size is say the max num of characters in any
            # word and the embedding size is 64 then you need
            # block_size * emb dim in the first layer as these embeddings 
            # get concatenated together
            nn.Linear(config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size) # decode to the output here
        )
        # save the embedding that is first initialised
        #print("\nPrint out initial emb\n")
        #emb_save_init = self.wte.weight.detach().numpy()
        #np.savetxt('emb_save_init.txt',emb_save_init, delimiter=',')
        

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None,isinference = False):
        device = idx.device
        # note the index comes in batches
        # of batch times num tokens
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        #emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        #hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        
        # basically have to add 2 columns to the start of idx for 0's of length batch
        idx = torch.cat([torch.zeros((b,2)),idx],axis=1).to(torch.int64)
        for i in range(t):
            inputs_ = idx[:,i:i+3]
            x = self.wte(inputs_)
            x = x.mean(axis=1)
            # emb already calculated per idx
            #xt = emb[:, i, :] # (b, n_embd)
            # now get the new hidden
            #ht = self.cell(xt, hprev) # (b, n_embd2)
            #hprev = ht
            #hiddens.append(ht)
            # only use the current token to predict the bext one
            
            # now form the input indicies, gather the embeddings and average
            # CBOW from the front only
            # could also try from either side
            '''
            if i == 0:
                inputs_ = torch.tensor([int(idx[0].numpy()),int(idx[0].numpy()),int(idx[0].numpy())])
                x = self.wte(inputs_)
                x = x.mean(axis=1)
                #xt = torch.cat([emb[:, i, :],emb[:, i, :],emb[:, i, :]],dim=1)
            elif i == 1:
                inputs_ = torch.tensor([int(idx[0].numpy()),int(idx[0].numpy()),int(idx[1].numpy())])
                x = self.wte(inputs_)
                x = x.mean(axis=1)
                #xt = torch.cat([emb[:, i, :],emb[:, i, :],emb[:, i, :]],dim=1)
            else:
                inputs_ = idx[i-2:i+1]
                x = self.wte(inputs_)
                x = x.mean(axis=1)
                #xt = torch.cat([emb[:, i, :],emb[:, i, :],emb[:, i, :]],dim=1)
            '''
            hiddens.append(x)
            #hiddens.append(xt)            
            #x = self.wte(inputs_)
            # note the tensor is 3D (batch,token_dim, embedding_dim)
            # with CBOW our goal is to average over the token embeddings to
            # end up with an average token - or in other words an average token embedding
            # hence we average over the token dim which in this case, with the batch, is axis=1
            # if it were strictly 2D it would be axis=0
            #x = x.mean(axis=1) # average the embeddings
            #x = self.linear(x)
                
            
            
                
            

        # decode the outputs over the prob distribution of the targets
        # essentially this turns a list into a tensor ready for a linear layer
        hidden = torch.stack(hiddens, 1) 
        logits = self.lm_head(hidden)
        #logits = self.mlp(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # the cross_entropy loss activates the logit
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # looks like here logits.view(-1, logits.size(-1)) will extract the last hidden state only

        return logits, loss




#%% -----------------------------------------------------------------------------
# MLP language model
# this uses the embeddings now but concatenates

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets 
        # inserted if encoding a token
        # before the beginning of the input sequence
        # config.vocab_size = 27 to account for 0 special character but we still
        # add ones again giving model.wte.weight.shape = torch.Size([28, 64])
        # obviously this is driven by the fact that the word indexes are 
        # 1 to 26
        # we can get special character 0 as well for start and end of name
        # also
        # we also get a character -1 as well which represents missing
        # hence need 28 rows in the embedding
        # once we start collecting embedding need to worry about how to process them
        # the other models only use config.vocab_size not +1
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) 
        self.mlp = nn.Sequential(
            # so if the block size is say the max num of characters in any
            # word and the embedding size is 64 then you need
            # block_size * emb dim in the first layer as these embeddings 
            # get concatenated together
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size) # decode to the output here
        )

    def get_block_size(self):
        return self.block_size
        #return 3
    # idx is an integer for the embedding matrix row
    def forward(self, idx, targets=None,isinference = False):

        # gather the word embeddings of the previous 3 chars????
        # no don't think it does that - there are no black size restrictions
        embs = []
        # this is the key part as we build embeddings over the chars in the word
        for k in range(self.block_size):
        #for k in range(block_size):
            # naturally processes over the batch
            # token embeddings of shape (b, tokens?, n_embd)
            # bear in mind early in the char prediction there may not be many
            # chars before the next char we need to predict
            
            # bear in mind the input idx here is over the whole block size
            # many of the values will eventually be filled with 0's as only if 
            # we predicted the last char of a max size word will the NN actually be full
            ####
            # if we wanted to restrict the network to say the last 3 tokens only we would do it 
            # at this point - the idea of zeroing out some would still apply.
            
            # say block_size * emb_dim even if many are the 0 tokens
            #print(idx)
            tok_emb = self.wte(idx) 
            #tok_emb = wte(idx)
            
            # these 2 next steps are the key
            # note that for each word we can make 
            # many predictions as we roll over the chars in the word
            # meaning ofcourse we can build many models for the same word
            # beginning predict char1
            # beginning + char1 predict char2 and so on
            # note that roll moves along the tensor of token integers
            idx = torch.roll(idx, 1, 1)
            # note that we then fill in the unfilled toekns with the 
            # special blank token
            idx[:, 0] = self.vocab_size # special <BLANK> token
            #idx[:, 0] = vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # concat all of the embeddings together and pass through an MLP
        # this must be aligned to the target
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        # check the mlp block above it takes flattenned embeddings
        logits = self.mlp(x)
        #print(x.shape)
        #print(x)
        #print(logits)
        #print(logits.view(-1, logits.size(-1)))
        #print(targets.view(-1))

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # note the ignore_index=-1 here means to Specifies a target value that is ignored
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

#%% -----------------------------------------------------------------------------
# Bigram language model
# I guess a bigram model is trainable - however it is learning logits or log(counts)
# which get passed through softmax via cross entropy loss
# REMEMBER the logit is log(p/1-p) which has domain [0,inf) and range (-inf,inf)
# which is fine in this case as count (exponential of the logit) has range [0,inf)
# ACTUALLY THIS IS REALLY THE NN TO LEARN THE BIGRAM FROM LECTURE 1
# it is not the bigram model as such since it is trainable

# It is a bit odd really as it is just calculating the probability of the next character given the previous character only.
# However it is doing it a strange way -- rather than directly calculating counts from the previous char and normalising to probabilities
# we build a model to logits (log counts) and then softmax
# Application to tennis ?? long range dependency may not be that important in tennis hence a trigram model may work.

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        # reads config but block size is not used
        n = config.vocab_size
        # trainable - NOTE VOCAB SIZE IS 27
        # it builds logits of size 27 x 27
        # which will ultimately capture the probilities
        # of each char from each char
        # think of it a 27 rows with a model over the 27 output (26 parameters in a categorical distribution)
        # in each row.
        # if we go to trigrams then it would be (27*27) rows or models to the output layer
        
        # this eventually represents an embedding over each char but specifically the logit of the next char
        # hence the unnormalised probability of the next char
        # I am not sure how much this embedding represents
        # however this is definitely an AR(1) model
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        # modify to 2 for trigrams
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None,isinference = False):

         # 'forward pass', lol - not much of a forward pass
         # NO ACTIVATION OR POST PROCESSING OF THE LOGITS
         # HENCE COMPLETELY UNNORMALISED AND NOT COUNTS EITHER AS CAN
         # BE NEGATIVE - hence log(counts)
         # 
         # gets the logits for the whole input at once say something like
         # tensor([ 0,  1, 15, 15,  1,  0,  0,  0,  0,  0,  0])
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        # remember target not aligned with x
        # tensor([ 1, 15, 15,  1,  0, -1, -1, -1, -1, -1, -1])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

#%% -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        # this is where black size really matters
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # forward the model to get the logits for the index in the sequence
        #logits, _ = model(idx_cond)
        # only for mytransformer
        logits, _ = model(idx_cond,isinference=True)

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        # the logits are NOT counts hence exponentiate then normalise
        probs = F.softmax(logits, dim=-1)
        # hence here is the generative part
        # either sample from the distribution or take the most likely element
        # even though the default is do_sample=False we normally pass
        # do_sample=True
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        # predicted value becomes next actual value
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

#%%

def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        # just need to end it at this point if it finds an ending value
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

#%%
@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        # no training here
        #logits, loss = model(X, Y)
        # this is going to be a problem as it will make mytransformer work but wrecks the mask
        logits, loss = model(X, Y,isinference=True) # won't work really as no mask means the evaluate will give unrealistic values
        
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

#%% -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

# inherits Dataset and adds extra methods not just the abstract ones

# initially the dataset is just initialised with all of the words and the chars it will need
# to build the vocab and numericalise
# however it only does the numericalisation once the __getitem__ is called by index
# or from a data loader
class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        # index 0 is not filled with a character - it will represent the start position
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        # all the possible characters and special 0 token
        # adding one is a key step here as the vocab size sets the dimension of the 
        # embedding vector and characters are 
        return len(self.chars) + 1 

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        # ix is a list not a tensor
        word = ''.join(self.itos[i] for i in ix)
        return word

    def decode_with_zeros(self, ix):
        # ix is a list not a tensor
        word = ''.join(self.itos[i] for i in ix if i>0)
        return word


    def __getitem__(self, idx):
    # when a word is retrieved it first gets encoded via the vocab
    # then the input and target tensors get initialised with zeros
    # then the encoded word gets placed in both the input vector and 
    # target vector however shifted back 1 for the target since we are predicting the next value
    # note the 0's remain in input and -1 for no predn in targ
    # say encodeing "anne"
    # x = tensor([ 0,  1, 15, 15,  1,  0,  0,  0,  0,  0,  0])
    # y = tensor([ 1, 15, 15,  1,  0, -1, -1, -1, -1, -1, -1])
    # batches of these are passed in to the model each time
    
        word = self.words[idx]
        ix = self.encode(word)
        # 0's to indicate the end of the word
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

#%%

def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    # w in words pull each char out of the string
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    #words = [w for w in words.split() if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters - set won't keep duplicates
    
    # calculating max word length becomes very important as it
    # ultimtely sets the max block length which are the max number of 
    # tokens we may use to predict the next token.
    # in some algorithms this is not used and a shorter black size is used
    # for example Bigram black size 1
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

#%%

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

#%% -----------------------------------------------------------------------------
if __name__ == '__main__':

    # can just run this and gather the defaults
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    # the -- means all of these arguments are optional
    # there are defaults if a parameter is not entered
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))
    type(vars(args)) # dict
    params = vars(args)
    # access the individual dict values in a few ways
    params['input_file']
    args.input_file
    

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    # input_file comes from the command line 
    print(f"The input file is {args.input_file}")
    train_dataset, test_dataset = create_datasets(args.input_file)
    train_dataset[0]
    train_dataset[0]
    train_dataset.itos
    train_dataset.stoi
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")
    input("\nEnter to continue")

    # use decode to decode ints from above
    print("The first vector from the training dataset is\n")
    print(train_dataset[0][0])
    print("\nThe decode is\n")
    print(train_dataset.decode(list([i for i in train_dataset[0][0].numpy() if i > 0])))
    input("\nEnter to continue")

    # init model via new dataclass class
    # must instatiate with vocab_size and block_size
    # the block size will be the max number of char in the words
    # vocab size is easy - number of char
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'mytransformer':
        model = MyTransformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'cbow':
            model = CBOW(config)
    elif args.type == 'mlpmine1':
        model = MLPMine1(config)
    elif args.type == 'mlpmine2':
        model = MLPMine2(config)
    elif args.type == 'mlpmine3':
        model = MLPMine3(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print("The model is\n")
    print(model)
    print("\nPrint out initial emb\n")
    if args.type in ['transformer','mytransformer']:
        emb = model.transformer.wte.weight
    elif args.type == 'bigram':
        emb = model.logits
    else:
        emb = model.wte.weight
    emb_save_initial = emb.detach().numpy()
    np.savetxt('emb_initial.txt',emb_save_initial, delimiter=',')

    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    #batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size)
    input(f"before getting started lets have a look at a dataset item used to train our {args.type} on\n {train_dataset[0]} \n based on the word {train_dataset.decode_with_zeros(list(train_dataset[0][0].numpy()))}")

    # training loop
    best_loss = None
    step = 0
    # set this manually if you want
    #args.max_steps = 10000
    
    # a baseline
    print(f" The baseline loss is {-np.log(1/27)}")
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        # gather logits don't really use them
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=16, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=16, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if step == args.max_steps/2:
            print("\nSave emb\n")
            if args.type in ['transformer','mytransformer']:
                emb = model.transformer.wte.weight
            elif args.type == 'bigram':
                emb = model.logits
            else:
                emb = model.wte.weight
            emb_save_half = emb.detach().numpy()
            np.savetxt('emb_save_half.txt',emb_save_half, delimiter=',')
        if args.max_steps >= 0 and step >= args.max_steps:
            break

# if writeout
if args.type in ['transformer','mytransformer']:
    emb = model.transformer.wte.weight
elif args.type == 'bigram':
    emb = model.logits
else:
    emb = model.wte.weight
emb_save = emb.detach().numpy()
np.savetxt('emb_final.txt',emb_save, delimiter=',')

import plot_the_embedding_prodn

datasets = ['emb_save_init.txt','emb_save_half.txt','emb_final.txt']
labels = []
labels = train_dataset.chars.copy()
labels.insert(0,'0')
print(labels)

for ds in datasets:
    plot_the_embedding_prodn.plot_2d_3d(ds,labels)

