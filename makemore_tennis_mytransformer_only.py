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

python makemore_spyder_tennis_shots.py -i tennis_shots_new_all_final_reduced.txt -o tennis_mlp --type mlp --max-steps 10000
python makemore_spyder_tennis_shots.py -i tennis_shots_new_all_final_reduced.txt -it <initial_token list e.g. a114,f39> -o tennis_mlp --type mlp --sample-only


run makemore_spyder_tennis_shots.py -i tennis_shots_new_all_final_reduced.txt -o tennis_mlp --type mlp --max-steps 10000
run makemore_spyder_tennis_shots.py -i tennis_shots_new_all_final_reduced.txt -it <initial_token e.g. a216> -o tennis_mlp --type mlp --sample-only
 
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

# NOTE IF YOU USE THIS NEED TO CHANGE THE CALL IN GENERATE AT LINE 1007 OR NEARBY DUE TO THE MASK
# change to this
# logits, _ = model(idx_cond,isinference=True) in generate
# also change evaluate()

class MyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # assign for later
        # no need for self here as enc_layer is not being passed to another method in the class
        # if a function is needed within a method and it is available outside of the class no problem
        # however if we define it within the class and it is needed in another method it must be put in self
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=config.n_embd,nhead=config.n_head, dim_feedforward=config.n_embd, batch_first=True)  # d_model divisible by nhead
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            trans_enc = torch.nn.TransformerEncoder(enc_layer,num_layers=config.n_layer))) # only about the number of layers
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
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        # just need to end it at this point if it finds an ending value
        if 'b3n@' in row:
            crop_index = row.index('b3n@') + 1
        elif 'b1n@' in row:
            crop_index = row.index('b1n@') + 1
        elif 'f19w@' in row:
            crop_index = row.index('f19w@') + 1
        elif 'b37w@' in row:
            crop_index = row.index('b37w@') + 1
        elif 'f1d@' in row:
            crop_index = row.index('f1d@') + 1
        elif 'b2n@' in row:
            crop_index = row.index('b2n@') + 1
        else:
            crop_index = len(row)
            
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
    model.eval() # still have to put the model into eval mode
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        #print(X)
        #input(f'The shape of the input is {X.shape}')
        # no training here
        # I doubt this is actually correct here!!
        # the model evaluate is taking a train and test batches
        # and passing though the model to get the error
        # the model should obviously still apply the mask as 
        # it will look ahead - need to fix this
        logits, loss = model(X, Y,isinference=True)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

#%% -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

# inherits Dataset and adds extra methods not just the abstract ones
class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
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
        ix = torch.tensor([self.stoi[w] for w in word.split(',')], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ','.join(self.itos[i] for i in ix)
        return word

    def decode_with_zeros(self, ix):
        # ix is a list not a tensor
        word = ','.join(self.itos[i] for i in ix if i>0)
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
    words = data.splitlines() # makes a list of individual "words"!
    # bear in mind here the fundamental word now is a 
    # comma delimited set of shot tokens
    # 'a125,q19,f18,b29,b3n@' NOT
    # say 'karen' which is a null-delimited set of char tokens
    # this means a fundamentally different approach to extracting the 
    # individual tokens which we will need to do for a vocab say
    # [w for w in "hello"] vs
    # [w for w in 'a125,q19,f18,b29,b3n@'.split(',')]
    # just becareful to understand whether working with a single word in the
    # form 'a125,q19,f18,b29,b3n@' or a list of such words
    # ['a125,q19,f18,b29,b3n@','a216,b29,f19,b17,f19,b39,b37,f28,f17,s28,f19,b1n@']
    # for which 
    # [w.split(',') for w in ['a125,q19,f18,b29,b3n@','a216,b29,f19,b17,f19,b39,b37,f28,f17,s28,f19,b1n@']]
    # is required
    
    #words = [w for w in words.split(',')]
    
    # can still do some processing on the individual words before vocab creation
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    
    words_to_make_vocab = [w.split(',') for w in words]
    #regular_list = [[1, 2, 3, 4], [5, 6, 7], [8, 9]]
    words_flat_list = [item for sublist in words_to_make_vocab for item in sublist]
    chars = sorted(list(set(words_flat_list)))
    # may only need this to clean up
    # w in words pull each char out of the string
    #words = [w.strip() for w in words.split(',')] # get rid of any leading or trailing white space
    #words = [w for w in words if w] # get rid of any empty strings
    #words = [w for w in words.split() if w] # get rid of any empty strings
    #chars = sorted(list(set(''.join(words)))) # all the possible characters - set won't keep duplicates
    
    # remember fundamentally the comma is NOT part  of the words just the 
    # method I use to seperate the individual shot tokens
    # hence as I pull an individual word out of the word list I must split it
    # into tokens
    max_word_length = max(len(w.split(',')) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(','.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.2)) # 20% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    # must change the decode and encode
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    # train_dataset.encode('f38,b28,f38')
    # train_dataset.decode([1,2,3])

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
    parser.add_argument('--input-file', '-i', type=str, default='tennis_shots_new_all_final_reduced.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='tennis_mytransformer', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='mytransformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
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
        if step > 0 and step % 100 == 0:
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
labels = train_dataset.chars.copy()
labels.insert(0,'0')
print(labels)
print(len(labels))

for ds in datasets:
    plot_the_embedding_prodn.plot_2d_3d(ds,labels)

