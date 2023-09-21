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

the best dataset I have at the moment is tennis_shots_new_all_final_reduced.txt - I may improve that at some stage.

python makemore_tennis_clean.py -i tennis_shots_new_all_final_reduced.txt -o tennis_mlpmine1 --type mlpmine1 --max-steps 10000
python makemore_tennis_clean.py -i tennis_shots_new_all_final_reduced.txt -it <initial_token list e.g. a114,f39> -o tennis_mlpmine1 --type mlpmine1 --sample-only


run makemore_tennis_clean.py -i tennis_shots_new_all_final_reduced.txt -o tennis_mlpmine1 --type mlpmine1 --max-steps 10000
run makemore_tennis_clean.py -i tennis_shots_new_all_final_reduced.txt -it <initial_token e.g. a216> -o tennis_mlpmine1 --type mlpmine1 --sample-only
 
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
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import spacy
import plot_the_embedding_prodn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#%%

#eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text


#%%

os.chdir('C:\gavin\software\python\pytorch\karpathy\makemore\makemore')
print(os.getcwd())

#%% -----------------------------------------------------------------------------

# dataclass is a new python feature
# basically a class that just stores data
# it is just a normal class without inheritance
# 
# special class no need for a def __init__(self) 
@dataclass
class ModelConfig:
    # Worth remembering that even though these values of block_size and vocab_size are None here
    # but they are always initialised by the particular model

    # maximum length of the sequence or number of shots in the point
    # also known as the context length - the information we are trying to learn from
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
# The plan here will be to 
# 1. change the fwd method to output the embeddings
# 2. instantiate the model 
# 3. load the weights pretrained weights
# 4. load in the data and output the embeddings


#%%

class PointsDataset(Dataset):

    def __init__(self, pts,vocab,max_pt_length):
        self.pts = pts
        self.vocab = vocab
        self.max_pt_length = max_pt_length
        self.stoi = vocab.get_stoi()
        self.itos = vocab.get_itos()

    def __len__(self):
        return len(self.pts)

    def contains(self, pt):
        return pt in self.pts

    def get_vocab_size(self):
        # vocab built with 0's
        return len(self.vocab) 

    def get_output_length(self):
        return self.max_pt_length + 1 # <START> token followed by words
        
    def encode(self, pt):
        #ix = torch.tensor([self.stoi[shot] for shot in pt.split(',')], dtype=torch.long)
        ix = torch.tensor(vocab(mytokenizer(pt)))
        return ix

    def decode(self, ix):
        pt = ','.join(self.itos[i] for i in ix)
        return pt

    def decode_with_zeros(self, ix):
        # ix is a list not a tensor
        pt = ','.join(self.itos[i] for i in ix if i>0)
        return pt


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

        pt = self.pts[idx]
        ix = self.encode(pt)
        # 0's to indicate the end of the word
        x = torch.zeros(self.max_pt_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_pt_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y


#%%

# will need a tokenizer
def mytokenizer(pt):
    return pt.strip().split(',')


mytokenizer('a216,f37,b3,s3,b3,s3,b3,s3,b1d@,<pad>')

# load the vocab from the transformer
vocab = torch.load(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_transformer\vocab.pt')

vocab(mytokenizer('a216,f37,b3,s3,b3,s3,b3,s3,b1d@,<pad>'))
len(vocab)
vocab.get_itos()

#%%

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
# Causal and not masked hence predicting the character at the end of the sequence

# This is just the multihead attention part only with an extra projection at the end.
# This is a long way from an actual encoder layer and then ofcourse we need to put several
# layers together
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    
    Note the number of heads to capture different contextual relationships
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # this is learnable about how it makes the q,k,value
        # in principle they should be the same - may end up being close
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        # notice how the output projection is of dimension n_embd
        # the transformer contextualises the embeddings but same dim
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        '''
        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.
        '''
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # the embedding only gets sent in once for the token so at this point we have to make the 
        # 3 copies for key, query and value
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2) # shapes (1*emb) * (emb * 3emb) = 1 * 3emb which is then split into 3 for q,k,v
        
        # now we make version for each head which is why embedding dim must be a multiple of # heads!!
        # note // returns the nearest integer 10//3 = 3 not 3.333333333
        
        # so for each head it essentially makes a smaller  version of each embedding vector rather than copies - works fine computationally effecient
        # this essentially just splits them for each head hence learns different things
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # after this
        # k.shape
        # Out[729]: torch.Size([2, 4, 16, 16])
        # batch of 2, 4 heads 16 tokens in each head each with an embedding in this case of 
        # emb_dim/# heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)  - hs embedding dim for that head
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # the query and the key combine to form the attention vectors
        # bunch of dot products that we later mask
        
        # att vectors are basically dim Token x Token which makes sense
        # since is it a set of dot products of each token embedding in the sequence with 
        # every other token embedding in the sequence INCLUDING itself hence T by T
        # it ultimately tells us how to weight the values/embeddings
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # the mask is also T by T as it is determining which of the attentions do we actually want to use
        # now to apply the mask
        # where the bias is 0 then replace the attention with -inf otherwise leave alone
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # now put the att values through a softmax
        # the softmax takes care of the -inf
        att = F.softmax(att, dim=-1)
        # now apply the masked attention to the values
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y.shape = orch.Size([2, 4, 16, 16])
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # y.shape = torch.Size([2, 16, 64])
        # output projection
        y = self.c_proj(y)
        return y


# really just forming the transformer block or layer by adding on the 
# extra things such as MLP etc
# Actual this just adds the components to the self-attention layer and forms a single encoder layer
# includes the residual connections and  

# this finishes the encoder layer comprising of the 2 sublayers
# self attn plus other stuff
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # why not just put in nn.Sequential and run all at once
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        # neat
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
    # note the 2 residual connections here
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
# note n_layer which tells us how many tranformer layers we want to put together.
class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # great example here of ModuleDict and ModuleList as compared to nn.Sequential
        # Module is the main building block and the base class of an neural network
        # class MyCNNClassifier(nn.Module):
        # but ModuleDict, ModuleList and Sequential are various containers of Modules
        # Sequential is easy as it is is a container of Modules that can be stacked together and run at the same time.
        # no need for a forward it is created automatically
        # the downside of Sequential is no flexibility in the application of the initiated modules in the fwd pass
        # when you need some more flexibility Module list is good. 
        # It is a container of Modules that can be stacked but then iterated through - hence not as restrictive as Sequential
        # Finally ModuleDict, being a Dict, allows you to access them by name and pick and choose as you need
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # whole encoder formed by multiple layers where each layer is self-attention plus the add on projections etc
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # the output of the transformer is dim n_embd which then
        # gets decoded to the output over the vocab
        # in theory this is not part of the transformer as it is decoding over the output distribution
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        # now we work through the layers for the total encoder
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits,x

#%%

# have to instantiate exactly the same model
n_layer = 4
n_head = 4
n_embd = 64
n_embd2 = 64
vocab_size=257
block_size=86

config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                   n_layer=n_layer, n_head=n_head,
                   n_embd=n_embd, n_embd2=n_embd2)

model = Transformer(config)
#vocab_size = train_dataset.get_vocab_size()
#block_size = train_dataset.get_output_length()

myweights = torch.load(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_transformer\model.pt')
#myweights = torch.load('C:\\gavin\\software\\python\\pytorch\\models\\xlmr.base.encoder.pt')
myweights.keys()
model.transformer.wte.weight
model.load_state_dict(myweights)
model.transformer.wte.weight

input_idx = torch.tensor(vocab(mytokenizer('a216,f37,b3,s3,b3,s3,b3,s3,b1d@,<pad>'))).view(1,-1)

input_idx.shape

output,api_endpoint = model(input_idx)

output.shape
api_endpoint.shape

api_endpoint = api_endpoint[0,:,:]
api_endpoint.shape

api_endpoint.mean(axis=0).shape

api_endpoint_average = api_endpoint.mean(axis=0)

torch.norm(api_endpoint_average - api_endpoint_average)

input_idx1 = torch.tensor(vocab(mytokenizer('a114,f18,f1,f1,f3*'))).view(1,-1)
input_idx2 = torch.tensor(vocab(mytokenizer('a124,f19,f1,f1,f3*'))).view(1,-1)
input_idx3 = torch.tensor(vocab(mytokenizer('a124,f19,f3,f3,b3*'))).view(1,-1)

output,api_endpoint1 = model(input_idx1)
api_endpoint1 = api_endpoint1[0,:,:]
api_endpoint1_average = api_endpoint1.mean(axis=0)

output,api_endpoint2 = model(input_idx2)
api_endpoint2 = api_endpoint2[0,:,:]
api_endpoint2_average = api_endpoint2.mean(axis=0)

output,api_endpoint3 = model(input_idx3)
api_endpoint3 = api_endpoint3[0,:,:]
api_endpoint3_average = api_endpoint3.mean(axis=0)

torch.norm(api_endpoint1_average - api_endpoint2_average)
torch.norm(api_endpoint1_average - api_endpoint3_average)

def points_dist(point1,point2):
    input_idx1 = torch.tensor(vocab(mytokenizer(point1))).view(1,-1)
    input_idx2 = torch.tensor(vocab(mytokenizer(point2))).view(1,-1)
    output,api_endpoint1 = model(input_idx1)
    api_endpoint1 = api_endpoint1[0,:,:]
    api_endpoint1_average = api_endpoint1.mean(axis=0)

    output,api_endpoint2 = model(input_idx2)
    api_endpoint2 = api_endpoint2[0,:,:]
    api_endpoint2_average = api_endpoint2.mean(axis=0)
    
    print(torch.norm(api_endpoint1_average - api_endpoint2_average))
    
points_dist('a114,f18,f1,f1,f3*','a114,f18,f1,f1,f3*')
points_dist('a114,f18,f1,f1,f3*','a124,f19,f1,f1,f3*')
points_dist('a114,f18,f1,f1,f3*','a124,f19,f3,f3,b3*')
points_dist('a124,f19,f1,f1,f3*','a124,f19,f3,f3,b3*')

points_dist('a225,f39,b2,f3,b1,f1,f1,f2,b3,b2,b1,f1,f1,f2,b3,b3,b1,f1,f3,s3,f1*','a216,f28,f1,f3,b3,b3,b3,b2,b3,s2,f3,b2,f1,f1,f1,f1,f3,b3,b2,f3,b3,b1,f1,f3,b2,y1,f3,b3,z1,f1,f2*')

points_dist('a216,b28,f3,b3,s3,y1n@','a224,b28,b1,f1,u3,s3,s3n#')

#%%
# embed all points with 5 or moreshots

api_embeddings = []

# start by reading all of the data into a list

#file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced_MLM.txt','r')
file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt','r')

all_pts = [line.strip() for line in file]
file.close()  

model.eval()
for pt in all_pts:
    input_idx = torch.tensor(vocab(mytokenizer(pt))).view(1,-1)
    output,api_endpoint = model(input_idx)
    api_endpoint = api_endpoint[0,:,:]
    api_endpoint_average = api_endpoint.mean(axis=0)
    api_embeddings.append(api_endpoint_average.detach().numpy())

model.train()
api_embeddings_np = np.array(api_embeddings)
np.savetxt('point_embeddings_transformer.txt',api_embeddings_np, delimiter=',')



plot_the_embedding_prodn.plot_2d_3d('point_embeddings_transformer.txt',all_pts)

plot_the_embedding_prodn.plot_2d_3d('point_embeddings_transformer.txt',list(range(len(all_pts))))

npdata_norm = (api_embeddings_np - api_embeddings_np.min(axis=0))/(api_embeddings_np.max(axis=0) - api_embeddings_np.min(axis=0))
kmeans_model = KMeans(n_clusters=8,random_state=0)
kmeans = kmeans_model.fit(npdata_norm)
kmeans_clusters = kmeans.predict(npdata_norm)

d = {'cluster': kmeans_clusters, 'points':all_pts}
clusters = pd.DataFrame(data=d)

clusters.sort_values(by='cluster',inplace=True)

#%%

# test above

# start by reading all of the data into a list

file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt','r')
all_pts = [line.strip() for line in file]
file.close()  
# build the vocab

def mytokenizer(pt):
    return pt.strip().split(',')

def yield_tokens(file_list):
     for line in file_list:
            # don't need a complex tokenizer here - yields a list via a simple tokenizer
            yield mytokenizer(line)

# specials go first
vocab = build_vocab_from_iterator(yield_tokens(all_pts),specials=["<pad>"])
#yield_tokens(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt'), 


torch.save(vocab, r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_transformer\vocab.pt')
# to load in later vocab = torch.load('vocab.pt')

itos = vocab.get_itos()
len(itos)

vocab(mytokenizer('a216,f37,b3,s3,b3,s3,b3,s3,b1d@,<pad>'))

max_pt_length = max(len(pt.split(',')) for pt in all_pts)


train_dataset = PointsDataset(all_pts,vocab, max_pt_length)
len(train_dataset)
train_dataset.contains('a114,f38,m2d#')
train_dataset.contains('a114,f38,m2d#!')
train_dataset.get_vocab_size()
len(itos)

train_dataset.get_output_length()
train_dataset.encode('a124,f17,f3,b3*')
train_dataset.decode([25, 64,  3, 61])
train_dataset.decode([25, 64,  3, 61,0])
train_dataset.decode_with_zeros([25, 64,  3, 61,0])

#%%

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


#%%
# Causal and not masked hence predicting the character at the end of the sequence

# This is just the multihead attention part only with an extra projection at the end.
# This is a long way from an actual encoder layer and then ofcourse we need to put several
# layers together
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    
    Note the number of heads to capture different contextual relationships
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # this is learnable about how it makes the q,k,value
        # in principle they should be the same - may end up being close
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        # notice how the output projection is of dimension n_embd
        # the transformer contextualises the embeddings but same dim
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        '''
        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.
        '''
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # the embedding only gets sent in once for the token so at this point we have to make the 
        # 3 copies for key, query and value
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2) # shapes (1*emb) * (emb * 3emb) = 1 * 3emb which is then split into 3 for q,k,v
        
        # now we make version for each head which is why embedding dim must be a multiple of # heads!!
        # note // returns the nearest integer 10//3 = 3 not 3.333333333
        
        # so for each head it essentially makes a smaller  version of each embedding vector rather than copies - works fine computationally effecient
        # this essentially just splits them for each head hence learns different things
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # after this
        # k.shape
        # Out[729]: torch.Size([2, 4, 16, 16])
        # batch of 2, 4 heads 16 tokens in each head each with an embedding in this case of 
        # emb_dim/# heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)  - hs embedding dim for that head
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # the query and the key combine to form the attention vectors
        # bunch of dot products that we later mask
        
        # att vectors are basically dim Token x Token which makes sense
        # since is it a set of dot products of each token embedding in the sequence with 
        # every other token embedding in the sequence INCLUDING itself hence T by T
        # it ultimately tells us how to weight the values/embeddings
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # the mask is also T by T as it is determining which of the attentions do we actually want to use
        # now to apply the mask
        # where the bias is 0 then replace the attention with -inf otherwise leave alone
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # now put the att values through a softmax
        # the softmax takes care of the -inf
        att = F.softmax(att, dim=-1)
        # now apply the masked attention to the values
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y.shape = orch.Size([2, 4, 16, 16])
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # y.shape = torch.Size([2, 16, 64])
        # output projection
        y = self.c_proj(y)
        return y

#%%

# really just forming the transformer block or layer by adding on the 
# extra things such as MLP etc
# Actual this just adds the components to the self-attention layer and forms a single encoder layer
# includes the residual connections and  

# this finishes the encoder layer comprising of the 2 sublayers
# self attn plus other stuff
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # why not just put in nn.Sequential and run all at once
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        # neat
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
    # note the 2 residual connections here
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

#%%

# note n_layer which tells us how many tranformer layers we want to put together.
class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # great example here of ModuleDict and ModuleList as compared to nn.Sequential
        # Module is the main building block and the base class of an neural network
        # class MyCNNClassifier(nn.Module):
        # but ModuleDict, ModuleList and Sequential are various containers of Modules
        # Sequential is easy as it is is a container of Modules that can be stacked together and run at the same time.
        # no need for a forward it is created automatically
        # the downside of Sequential is no flexibility in the application of the initiated modules in the fwd pass
        # when you need some more flexibility Module list is good. 
        # It is a container of Modules that can be stacked but then iterated through - hence not as restrictive as Sequential
        # Finally ModuleDict, being a Dict, allows you to access them by name and pick and choose as you need
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # whole encoder formed by multiple layers where each layer is self-attention plus the add on projections etc
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # the output of the transformer is dim n_embd which then
        # gets decoded to the output over the vocab
        # in theory this is not part of the transformer as it is decoding over the output distribution
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        # now we work through the layers for the total encoder
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
class MyRNN(nn.Module):
    # instantiate with config 
    def __init__(self, config):
        super().__init__()
        # the config object gets passed into the constructor which means it is available to use
        # in the constructor only
        # to ensure the config values are available to ther methods in the class
        # then assign the values to self
        # self. means they are available to other
        # methods on the class but not accessed outside of the class
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # get the starting token embedding and starting hidden state
        # the starting hidden state
        #self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) 
        # token embeddings table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0) 
        #if cell_type == 'rnn':
        #    self.cell = RNNCell(config)
        #elif cell_type == 'gru':
        #    self.cell = GRUCell(config)
        # self.rnn = nn.RNN(config.n_embd,config.n_embd,batch_first=True)
        self.rnn = nn.LSTM(config.n_embd,config.n_embd,batch_first=True)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        # note that the config is not passed in here but self is hence 
        # any attribute assigned to self is available
        return self.block_size 

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        ## hprev = self.start.expand((b, -1)) # expand out the batch dimension
        # at this point we gather the hidden layers from each point in the 
        # sequence as it is from each of these h vectors we predict the next token
        ##hiddens = []
        ##for i in range(t):
        ##    xt = emb[:, i, :] # (b, n_embd)
            # now get the new hidden
        ##    ht = self.cell(xt, hprev) # (b, n_embd2)
        ##    hprev = ht
        ##    hiddens.append(ht)

        # decode the outputs
        # (b, t, n_embd2) - batch, timesteps,num units in hidden layer 
        # - ultimate extract the final hidden state at the final timestep for each batch
        # maybe not in this case as it is predicting at each timestep
        # definitely not in this case we are keeping the hidden representation at each sequence
        # as that is being used to
        # 1. capture the latent structure TO THAT POINT
        # 2. capture the representation at that point to predict the aligned token
        #hidden = torch.stack(hiddens, 1)
        hidden, last = self.rnn(emb)        
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
        self.wte = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0) 
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

    def forward(self, idx, targets=None):
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
        self.wte = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0) 
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

    def forward(self, idx, targets=None):
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
        self.wte = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0) 
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

    def forward(self, idx, targets=None):
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
        #logits = self.lm_head(hidden)
        logits = self.mlp(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # the cross_entropy loss activates the logit
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # looks like here logits.view(-1, logits.size(-1)) will extract the last hidden state only

        return logits, loss

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        # trainable - NOTE VOCAB SIZE IS 27
        # it builds logits of size 27 x 27
        # which will ultimately capture the probilities
        # of each char from each char
        # think of it a 27 rows with a model over the 27 output (26 parameters in a categorical distribution)
        # in each row.
        # if we go to trigrams then it would be (27*27) rows or models to the output layer
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol - not much of a forward pass
         # NO ACTIVATION OR POST PROCESSING OF THE LOGITS
         # HENCE COMPLETELY UNNORMALISED AND NOT COUNTS EITHER AS CAN
         # BE NEGATIVE - hence log(counts)
         # 
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
#%% -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None,startidx=[-1]):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for count in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # only for mytransformer
        #logits, _ = model(idx_cond,isinference=True)
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
        if do_sample:
            if ((count == 0) & (startidx[0]>0)):
                #idx_next = torch.tensor([6,3])*torch.ones(idx.size(),dtype=torch.int64)
                # idx_next = startidx*torch.ones(idx.size(),dtype=torch.int64)
                idx_next = torch.tensor(startidx) * torch.ones((idx.size(0),len(startidx)),dtype=torch.int64)
            else:    
                idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

#%%

def print_samples(num=10,initial_token=[-1]):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    #X_init = torch.ones(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    # update this to a list of token indicies
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True,startidx=initial_token).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        # note the first value will be the start token don't need that
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        # just need to end it at this point if it finds an ending value
        # above all wasted standard crop index is fine
        # as soon as a 0 is predictied - the end that is it
        crop_index_old = row.index(0) if 0 in row else len(row)
        row = row[:crop_index_old]
        #row = row[:(crop_index-1)]
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
        #print(X)
        #input(f'The shape of the input is {X.shape}')
        # no training here
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss


#%%

def create_datasets(input_file):

    file = open(input_file,'r')
    all_pts = [line.strip() for line in file]
    file.close()  
    # build the vocab

    def mytokenizer(pt):
        return pt.strip().split(',')

    def yield_tokens(file_list):
         for line in file_list:
                # don't need a complex tokenizer here - yields a list via a simple tokenizer
                yield mytokenizer(line)

    # specials go first
    vocab = build_vocab_from_iterator(yield_tokens(all_pts),specials=["<pad>"])
    #yield_tokens(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt'), 
     
    itos = vocab.get_itos()
    len(itos)

    vocab(mytokenizer('a216,f37,b3,s3,b3,s3,b3,s3,b1d@,<pad>'))

    max_pt_length = max(len(pt.split(',')) for pt in all_pts)


    train_dataset = PointsDataset(all_pts,vocab, max_pt_length)

    test_set_size = int(len(all_pts) * 0.2) # 20% of the training set, or up to 1000 examples
    rp = torch.randperm(len(all_pts)).tolist()
    train_words = [all_pts[i] for i in rp[:-test_set_size]]
    test_words = [all_pts[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    # must change the decode and encode
    train_dataset = PointsDataset(train_words, vocab, max_pt_length)
    test_dataset = PointsDataset(test_words, vocab, max_pt_length)

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
    parser.add_argument('--initial-token', '-it', type=str, default=None, help="specify token of first shot")
    args = parser.parse_args()
    input(f'\n the inputs are {vars(args)}.\nEnter to continue')
    type(vars(args)) # dict
    params = vars(args)
    # note args is a dict so it you want to run it with inputs as below modify params
    # run makemore_spyder_tennis_shots.py -i tennis_shots_new_all_final_reduced.txt -o tennis_mlp --type mlp --max-steps 10000
    # access the individual dict values in a few ways
    params['input_file']
    args.input_file
    
    # make modifications if you wish
    # only uncomment if you run within spyder
    #args.input_file = 'tennis_shots_new_all_final_reduced.txt'
    #args.work_dir = 'tennis_mlp'
    #args.type = 'mlp'
    #args.max_steps = 10000
    #args



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
    print('\nCheck vocab and block size.\nEnter to continue')
    input("\nEnter to continue")

    # use decode to decode ints from above
    print("The first vector from the training dataset is\n")
    print(train_dataset[0][0])
    print("\nThe decode is\n")
    print(train_dataset.decode(list([i for i in train_dataset[0][0].numpy() if i > 0])))
    input("\nEnter to continue")

    # use decode to decode ints from above
    train_dataset.decode(list([i for i in train_dataset[0][0].numpy() if i > 0]))

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
    elif args.type == 'myrnn':
        model = MyRNN(config)
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
        if args.initial_token == None:
            print_samples(num=50)
        else:
            # numericalise the initial token here before passing in
            # to add another token do here
            # how to eventually use multiple inputs via a list
            # torch.tensor([3,4]) * torch.ones((10,2),dtype=torch.int64)
            # eventually pass in as a list and use torch.tensor([3,4]) * torch.ones((10,2),dtype=torch.int64)
            initial_token_list = [train_dataset.stoi[val] for val in args.initial_token.split(',')]
            #print_samples(num=50,initial_token=train_dataset.stoi[args.initial_token])
            print_samples(num=50,initial_token=initial_token_list)
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
    # set this  manually if you want
    # args.max_steps = 10000
    
    # a baseline get_vocab_size
    input(f" The baseline loss is {-np.log(1/train_dataset.get_vocab_size())}\nEnter to continue")
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
            #print('\nNo Sample \n')

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
            np.savetxt('emb_half.txt',emb_save_half, delimiter=',')
        
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

# testing - if need new labels
#train_dataset, test_dataset = create_datasets('tennis_shots_new_all_final_reduced.txt')

datasets = ['emb_initial.txt','emb_half.txt','emb_final.txt']
labels = []
labels = train_dataset.itos[1:].copy()
labels.insert(0,'0')
print(labels)
print(len(labels))

for ds in datasets:
    plot_the_embedding_prodn.plot_2d_3d(ds,labels)
