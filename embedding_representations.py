# usage
# python embedding_representations.py <model_directory>
# where <model_directory> is the path to a trained transformer model with a model.pt file
# containing the trained weights
# example
# python embedding_representations.py .\tennis_gpt\transformer

"""
Before running this ensure the tennis_gpt model has been run for the 
transformer model, as this only works for transformer models, such as
python tennis_gpt_colab.py -i tennis_shot_data.txt -o <transfomer_dir>  --max-steps 10000 --type transformer
ensure the directory argument points to a directory with this output
"""


from tennis_gpt import *
import plot_the_embedding_prodn
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import torch

# Load the model directory from the command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <directory_containing_model_pt>")
    sys.exit(1)

model_directory = sys.argv[1]

#%%
# load the vocab from this directory - already built
vocab = torch.load('./master_vocab/vocab.pt')
# load the weights
# the transformer_final model is NOT finetuned but was the base model for the fine tuning
#myweights = torch.load(r'./transformer_final/model.pt')
myweights = torch.load(os.path.join(model_directory, 'model.pt'))

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
        #print(output)
    # remove the batch dim    
    api_endpoint = api_endpoint[0,:,:]

    # Two choices for capturing the full point embedding
    #
    # 1. Average over all of the token embeddings in the point
    api_endpoint_average = api_endpoint.mean(axis=0)

    # 
    # 2. Use the embedding of the last token?
    # this most likely will not give an interesting set of representaions as 
    # the next token for all of them will be the end of point token
    # so all representaions are much the same.
    # It doesn't really capture the point representation
    # but you can try it
    #api_endpoint_average = api_endpoint[-1,:]
    

    api_embeddings.append(api_endpoint_average.detach().numpy())

# create a little vector database of embeddings
api_embeddings_np = np.array(api_embeddings)
# Save the embeddings to the same directory as model.pt
embeddings_filename = 'point_embeddings_transformer.txt'
np.savetxt(os.path.join(model_directory, embeddings_filename), api_embeddings_np, delimiter=',')

# Plot the embeddings using the saved file
out_path = './'
plot_the_embedding_prodn.plot_embedding(os.path.join(model_directory, embeddings_filename), all_pts, out_path=model_directory, transparent=True)

