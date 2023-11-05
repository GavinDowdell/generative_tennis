"""
Before running this ensure the tennis_gpt model has been run for the 
transformer model, as below, with the output in the directory tennis_gpt
python tennis_gpt.py -i tennis_shot_data.txt -o tennis_gpt --type transformer --max-steps 10000

"""

from tennis_gpt import *
import plot_the_embedding_prodn
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

#%%

# load the vocab
vocab = torch.load('./tennis_gpt/vocab.pt')

# check the vocab works
# must put in <pad> for the start of the point
# otherwise the positional encoding is incorrect
test_point = '<pad>,a114,f18,f1' 

# The sequence makes sense as expecting a forehand shot here such as 
# a114 is a first serve to the forehand side out wide hence must be 
# followed by a forehand
# f1,f2,f3 as the next probable shot
mytokenizer(test_point)

# Can use the vocab with the tokenizer
vocab(mytokenizer(test_point))
# check the reverse map works
vocab.lookup_tokens(vocab(mytokenizer(test_point)))

# Now Load the transformer weights, not into the model yet
myweights = torch.load(r'./tennis_gpt/model.pt')
myweights.keys()
myweights['lm_head.weight']
# will transform any 64 dim embedding to a representation over the embedding
myweights['lm_head.weight'].shape
# Will have to instantiate exactly the same model that was loaded in
n_layer = 4
n_head = 4
n_embd = 64
n_embd2 = 64
vocab_size=257
block_size=86

config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                   n_layer=n_layer, n_head=n_head,
                   n_embd=n_embd, n_embd2=n_embd2)

# behaves like a class not a dict as ModelConfig is a data class
config.n_layer

#%%
# will first have to restore the model
# Instantiate the model which will give the model the architecture with random weights
model = Transformer(config)
# can use the model at this point however it is just random weights
out = model(torch.tensor(vocab(mytokenizer(test_point))).reshape(1,-1),return_embedding=False)
# collect the logit
type(out[0])
out[0].shape
# the final output layer
out[0][0,-1,:].shape
# get the max logit random weights here
vocab.lookup_tokens([out[0][0,-1,:].argmax()])
vocab.lookup_tokens(out[0][0,-1,:].topk(5)[1].tolist())
# Now load the weights into the model
model.transformer.wte.weight
model.load_state_dict(myweights)
model.transformer.wte.weight
out = model(torch.tensor(vocab(mytokenizer(test_point))).reshape(1,-1),return_embedding=True)
vocab.lookup_tokens([out[0][0,-1,:].argmax()])
vocab.lookup_tokens(out[0][0,-1,:].topk(5)[1].tolist())

# embeddings - 2 choices
# mean
out[1][0,:,:].mean(axis=0).shape
# final
out[1][0,2,:].shape

# Distance function
# can choose for a point to average the embeddings or
# take the last embedding which is a transformer autoregressive
# contextualised embedding ready to predict the next token
# hence captures the full meaning of the sequence

def points_dist(point1,point2,mean=True):
    input_idx1 = torch.tensor(vocab(mytokenizer(point1))).view(1,-1)
    input_idx2 = torch.tensor(vocab(mytokenizer(point2))).view(1,-1)
    output,api_endpoint1 = model(input_idx1,return_embedding=True)
    api_endpoint1 = api_endpoint1[0,:,:]
    # don;t have to average here - could take the last representation or average
    if mean:
        api_endpoint1_average = api_endpoint1.mean(axis=0)
    if not mean:
        # use this if the representation you want is the 
        # final representation
        api_endpoint1_average = api_endpoint1[-1,:]
        

    output,api_endpoint2 = model(input_idx2,return_embedding=True)
    api_endpoint2 = api_endpoint2[0,:,:]
    api_endpoint2_average = api_endpoint2.mean(axis=0)
    if mean:
        api_endpoint2_average = api_endpoint2.mean(axis=0)
    if not mean:
        api_endpoint2_average = api_endpoint2[-1,:]
    
    print(torch.norm(api_endpoint1_average - api_endpoint2_average))
    print(1-torch.cosine_similarity(api_endpoint1_average.view(1,-1),api_endpoint2_average.view(1,-1)))
    
points_dist('<pad>,a114,f18,f1,f1,f3*','<pad>,a114,f18,f1,f1,f3*')
points_dist('<pad>,a114,f18,f1,f1,f3*','<pad>,a114,f18,f1,f1,f3*',False)
points_dist('<pad>,a114,f18','<pad>,a214,f18')
points_dist('<pad>,a114,f18','<pad>,a214,f18',False)
points_dist('<pad>,a114,f18','<pad>,a214,f19')
points_dist('<pad>,a114,f18','<pad>,a214,f19',False)
# cosine distance of the final representation is probably preferred?

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
    output,api_endpoint = model(input_idx,return_embedding=True)
    api_endpoint = api_endpoint[0,:,:]
    # either average the embeddings or take the last embedding
    # the last embedding is ok for gpt model
    # however if we are trying to represent the whole point 
    # need to average over all embeddings as the final embedding
    # is the end of the point
    api_endpoint_average = api_endpoint.mean(axis=0)
    #api_endpoint_average = api_endpoint[-1,:]
    api_embeddings.append(api_endpoint_average.detach().numpy())

model.train()
# create a little vector database of embeddings
api_embeddings_np = np.array(api_embeddings)
np.savetxt('point_embeddings_transformer.txt',api_embeddings_np, delimiter=',')

# can plot as well
plot_the_embedding_prodn.plot_embedding('point_embeddings_transformer.txt',all_pts,transparent=True)

#%%
# Retrieval closest points from a test point
# from this find the nearest point to the test point

test_point = '<pad>,a114,f1n#'
test_point = '<pad>,a214,b28,f3,b3n#' # not a point in the dataset
test_point = '<pad>,a214,b28,f3,s2d#' # a point


out = model(torch.tensor(vocab(mytokenizer(test_point))).reshape(1,-1),return_embedding=True)
# check the logits of the test_point are behaving correctly
# the logit is over the shot vocab hence should be a good 
# representation of the next shot
# should be <pad> as the point is over
vocab.lookup_tokens(out[0][0,-1,:].topk(10)[1].tolist())
# from the second last shot
vocab.lookup_tokens(out[0][0,-2,:].topk(10)[1].tolist())

# pick the representation that matches how the dataset was represented
# IF IT IS THE WHOLE POINT THEN THE FINAL EMBEDDING DOESN'T MAKE SENSE
#average
test_embed = out[1][0,:,:].mean(axis=0).detach().numpy()
#final representation
test_embed = out[1][0,-1,:].detach().numpy()

#%%
# Euclidean distances
distances = np.linalg.norm(api_embeddings_np-test_embed, axis=1)
min_index = np.argmin(distances)
print(min_index)
print(distances[min_index])
print(api_embeddings_np[min_index])
print(all_pts[min_index])

indices_of_smallest_values = np.argsort(distances)

for i in indices_of_smallest_values[:20]:
    print(all_pts[i])
#%%  
# Cosine distance is defined as 1.0 minus the cosine similarity.
distances = cosine_distances(api_embeddings_np,test_embed.reshape(1, -1))
distances = distances.reshape(-1,)
min_index = np.argmin(distances)
min_index
print(min_index)
print(distances[min_index])
print(api_embeddings_np[min_index])
print(all_pts[min_index])

indices_of_smallest_values = np.argsort(distances)

for i in indices_of_smallest_values[:20]:
    print(all_pts[i])

#%%





