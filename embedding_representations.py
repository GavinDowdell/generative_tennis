"""
Before running this ensure the tennis_gpt model has been run for the 
transformer model, as below, with the output in the directory tennis_gpt
python tennis_gpt.py -i tennis_shot_data.txt -o tennis_gpt --type transformer --max-steps 10000

"""

from tennis_gpt import *
import plot_the_embedding_prodn
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import torch

#%%
# load the vocab
vocab = torch.load('./tennis_gpt/vocab.pt')
# load the weights
myweights = torch.load(r'./tennis_gpt/model.pt')
config = ModelConfig(vocab_size=len(vocab), block_size=86,
                   n_layer=4, n_head=4,
                   n_embd=64, n_embd2=64)


#%%
# will first have to restore the model
# Instantiate the model which will give the model the architecture with random weights
model = Transformer(config)
# load weights
model.load_state_dict(myweights)
#%%

# embed all points, cluster and plot
api_embeddings = []

file = open(r'tennis_shot_data_unique.txt','r')

all_pts = [line.strip() for line in file]
file.close()  

model.eval()

for pt in all_pts:
    input_idx = torch.tensor(vocab(mytokenizer(pt))).view(1,-1)
    # put a zero at the start for <pad>
    # otherwise the positional encoding will be wrong
    input_idx = torch.cat((torch.tensor(0).view(1,1),input_idx),axis=1)
    with torch.no_grad():
        output,api_endpoint = model(input_idx,return_embedding=True)
    api_endpoint = api_endpoint[0,:,:]
    api_endpoint_average = api_endpoint.mean(axis=0)
    #api_endpoint_average = api_endpoint[-1,:]
    api_embeddings.append(api_endpoint_average.detach().numpy())

# create a little vector database of embeddings
api_embeddings_np = np.array(api_embeddings)
np.savetxt('point_embeddings_transformer.txt',api_embeddings_np, delimiter=',')

# can plot as well
plot_the_embedding_prodn.plot_embedding('point_embeddings_transformer.txt',all_pts,transparent=True)



