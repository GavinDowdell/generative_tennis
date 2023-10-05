"""
Before running this ensure the tennis_gpt model has been run for the 
transformer model, as below, with the output in the directory tennis_gpt
python tennis_gpt.py -i tennis_shots_new_all_final_reduced.txt -o tennis_gpt --type transformer --max-steps 10000

"""


# will clobber the namespace but I know the origin so that's ok
from tennis_gpt import *

mytokenizer('a112,f38,f3')

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

# behaves like a class not a dict
config.n_layer


# will first have to restore the model
# 3 steps
# 1. Instantiate the model which will give the model the architecture with random weights
model = Transformer(config)
# 2. Load the weights into a file - not into the model
myweights = torch.load(r'./tennis_gpt/model.pt')
myweights.keys()
myweights['lm_head.weight']
# will transform any 64 dim embedding to a representation over the embedding
myweights['lm_head.weight'].shape
# could actually use these weights directly if you want
emb_test = torch.rand((64,2))
test_out = myweights['lm_head.weight'] @ emb_test
test_out.shape
test_out.max(axis=0)


# 3. Now load the weights into the model
model.transformer.wte.weight
model.load_state_dict(myweights)
model.transformer.wte.weight