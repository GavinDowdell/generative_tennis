"""
Before running this ensure the tennis_gpt model has been run for the 
transformer model, as below, with the output in the directory tennis_gpt
python tennis_gpt.py -i tennis_shots_new_all_final_reduced.txt -o tennis_gpt --type transformer --max-steps 10000

"""


# will clobber the namespace but I know the origin so that's ok
from tennis_gpt import *
# must load a vocab
vocab = torch.load('./tennis_gpt/vocab.pt')
test_point = 'a114,f18,f1' # expecting at f1,f2,f3 here as the next probable shot
mytokenizer(test_point)

# 1. Can use the tokenizer
vocab(mytokenizer(test_point))
vocab.lookup_indices(mytokenizer(test_point))
vocab.lookup_tokens(vocab(mytokenizer(test_point)))

# 2. Load the transformer weights, not into the model yet
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
test_out.argmax(axis=0)
test_out.topk(5)[1]
test_out[:,0].topk(5)
# have to instantiate exactly the same model that was loaded in
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


# will first have to restore the model
# 3 steps
# 1. Instantiate the model which will give the model the architecture with random weights
model = Transformer(config)

# can use the model at this point however it is just random weights
out = model(torch.tensor(vocab(mytokenizer(test_point))).reshape(1,-1),return_embedding=False)
# collect the logit
type(out[0])
out[0].shape
out[0][0].shape
out[0][0,2,:].shape

# get the max logit random weights here
vocab.lookup_tokens([out[0][0,2,:].argmax()])

out = model(torch.tensor(vocab(mytokenizer(test_point))).reshape(1,-1),return_embedding=True)
out[0].shape
out[1].shape
vocab.lookup_tokens([out[0][0,2,:].argmax()])

# 2. Load the weights which I have already done

# 3. Now load the weights into the model
model.transformer.wte.weight
model.load_state_dict(myweights)
model.transformer.wte.weight

out = model(torch.tensor(vocab(mytokenizer(test_point))).reshape(1,-1),return_embedding=True)
vocab.lookup_tokens([out[0][0,2,:].argmax()])
vocab.lookup_tokens(out[0][0,2,:].topk(5)[1].tolist())

# embeddings
# mean
out[1][0,:,:].mean(axis=0).shape
# final
out[1][0,2,:].shape

# develop and test a distance function

def points_dist(point1,point2,mean=True):
    input_idx1 = torch.tensor(vocab(mytokenizer(point1))).view(1,-1)
    input_idx2 = torch.tensor(vocab(mytokenizer(point2))).view(1,-1)
    output,api_endpoint1 = model(input_idx1,return_embedding=True)
    api_endpoint1 = api_endpoint1[0,:,:]
    # don;t have to average here - could take the last representation or average
    if mean:
        api_endpoint1_average = api_endpoint1.mean(axis=0)
    if not mean:
        api_endpoint1_average = api_endpoint1[-1,:]
        

    output,api_endpoint2 = model(input_idx2,return_embedding=True)
    api_endpoint2 = api_endpoint2[0,:,:]
    api_endpoint2_average = api_endpoint2.mean(axis=0)
    if mean:
        api_endpoint2_average = api_endpoint2.mean(axis=0)
    if not mean:
        api_endpoint2_average = api_endpoint2[-1,:]
    
    print(torch.norm(api_endpoint1_average - api_endpoint2_average))
    print(torch.cosine_similarity(api_endpoint1_average.view(1,-1),api_endpoint2_average.view(1,-1)))
    
points_dist('a114,f18,f1,f1,f3*','a114,f18,f1,f1,f3*')
points_dist('a114,f18,f1,f1,f3*','a114,f18,f1,f1,f3*',False)
points_dist('a114,f18','a214,f19')
points_dist('a114,f18','a214,f18',False)
points_dist('a114,f18','a214,f19')
points_dist('a114,f18','a214,f19',False)

points_dist('a114,f18,f1,f1,f3*','a124,f19,f1,f1,f3*')
points_dist('a114,f18,f1,f1,f3*','a124,f19,f3,f3,b3*')
points_dist('a124,f19,f1,f1,f3*','a124,f19,f3,f3,b3*')

points_dist('a225,f39,b2,f3,b1,f1,f1,f2,b3,b2,b1,f1,f1,f2,b3,b3,b1,f1,f3,s3,f1*','a216,f28,f1,f3,b3,b3,b3,b2,b3,s2,f3,b2,f1,f1,f1,f1,f3,b3,b2,f3,b3,b1,f1,f3,b2,y1,f3,b3,z1,f1,f2*')

points_dist('a216,b28,f3,b3,s3,y1n@','a224,b28,b1,f1,u3,s3,s3n#')

#%%

# embed all points with 5 or moreshots
api_embeddings = []

file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt','r')

all_pts = [line.strip() for line in file]
file.close()  

model.eval()
for pt in all_pts:
    input_idx = torch.tensor(vocab(mytokenizer(pt))).view(1,-1)
    output,api_endpoint = model(input_idx,return_embedding=True)
    api_endpoint = api_endpoint[0,:,:]
    api_endpoint_average = api_endpoint.mean(axis=0)
    api_endpoint_average = api_endpoint[-2,:]
    api_embeddings.append(api_endpoint_average.detach().numpy())

model.train()
api_embeddings_np = np.array(api_embeddings)
np.savetxt('point_embeddings_transformer.txt',api_embeddings_np, delimiter=',')


