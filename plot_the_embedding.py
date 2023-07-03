# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 23:38:51 2023

@author: gavin
"""


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly
import numpy as np

# train_dataset still around
labels = []
labels = train_dataset.chars.copy()
labels.insert(0,'0')
labels

# embeddings come from the makemore code
# which model they come from matters and the depth of the model
# for example if there are more layers after the embedding then
# the meaning of them may vary

dataset = 'emb_save_init.txt'
dataset = 'emb_save_half.txt'
dataset = 'emb.txt'


data = pd.read_csv(dataset,header=None)
data.head()
data.shape

# before thinking about dimension reduction need to consider the mean
# of the features particularly the variance - which matters when it comes
# to whether to scale the data - for featores that have completely different
# means then scaling really matters
data.mean()
data.var()
data.var().plot()
data.mean().plot()
plt.plot()

# may want to do pca25 and then tsne or straight to pca3 or 2
pca = PCA(25)
data_pca = pca.fit_transform(data.values)

# try for lower dim directly
pca3 = PCA(3)
data_pca3 = pca3.fit_transform(data.values)
pca2 = PCA(2)
data_pca2 = pca2.fit_transform(data.values)

# make a dataframe with labels
d = {'letter': labels, 'pca1': data_pca2[:,0], 'pca2': data_pca2[:,1]}
df_pca2 = pd.DataFrame(data=d)

d = {'letter': labels, 'pca1': data_pca3[:,0], 'pca2': data_pca3[:,1], 'pca3': data_pca3[:,2]}
df_pca3 = pd.DataFrame(data=d)

# try some plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(data_pca3[:,0],data_pca3[:,1],data_pca3[:,2])
plt.show()

    
# Set the figure size
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Scatter plot
ax = df_pca2.plot.scatter(x='pca1', y='pca2', alpha=0.5)

# Annotate each data point
for i, txt in enumerate(df_pca2.letter):
   ax.annotate(txt, (df_pca2.pca1.iat[i]+0.05, df_pca2.pca2.iat[i]))

plt.show()



fig = px.scatter(df_pca2, x="pca1", y="pca2", text="letter", size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='Letter embedding1'
)

fig.show()



plotly.offline.plot(fig,filename='plotly_express1.html')


fig = px.scatter_3d(df_pca3, x="pca1", y="pca2", z = "pca3" , text="letter", size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='Letter embedding2'
)

fig.show()



plotly.offline.plot(fig,filename='plotly_express2.html')


tsne = TSNE(n_components=2)
# we fit tsne to the pca data
embedding = tsne.fit_transform(data_pca)

embedding_df = pd.DataFrame(embedding, columns=["x", "y"])

embedding_df.index = np.squeeze(labels)

embedding_df.to_csv('emb_tsne.txt')


fig = px.scatter(embedding_df, x="x", y="y", text=embedding_df.index, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='Letter embedding3'
)

fig.show()



plotly.offline.plot(fig,filename='plotly_express3.html')


tsne = TSNE(n_components=3)
# we fit tsne to the pca data
embedding = tsne.fit_transform(data_pca)

embedding_df = pd.DataFrame(embedding, columns=["x", "y", "z"])

embedding_df.index = np.squeeze(labels)

embedding_df.to_csv('emb_tsne3d.txt')


fig = px.scatter_3d(embedding_df, x="x", y="y", z = "z" , text=embedding_df.index, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='Letter embedding4'
)

fig.show()



plotly.offline.plot(fig,filename='plotly_express4.html')

tsne = TSNE(n_components=2)
# we fit tsne to the pca data
embedding = tsne.fit_transform(data.values)

embedding_df = pd.DataFrame(embedding, columns=["x", "y"])

embedding_df.index = np.squeeze(labels)

embedding_df.to_csv('emb_tsne.txt')


fig = px.scatter(embedding_df, x="x", y="y", text=embedding_df.index, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='Letter embedding3'
)

fig.show()

plotly.offline.plot(fig,filename='plotly_express5.html')


embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
embeddings_norm.shape
