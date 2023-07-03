import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly
import numpy as np



# embeddings come from the makemore code
# which model they come from matters and the depth of the model
# for example if there are more layers after the embedding then
# the meaning of them may vary

def plot_2d_3d(dataset,labels):
    data = pd.read_csv(dataset,header=None)
    #data.var().plot()
    #data.mean().plot()
    #plt.plot()
    #plt.show()
    
    # scale
    npdata = data.values
    # could also use MinMaxScaler
    npdata_norm = (npdata - npdata.min(axis=0))/(npdata.max(axis=0) - npdata.min(axis=0))
    #plt.plot(npdata_norm.var(axis=0))
    #plt.plot(npdata_norm.mean(axis=0))
    #plt.show()
    
    # try for lower dim directly
    pca3 = PCA(3)
    data_pca3 = pca3.fit_transform(npdata_norm)
    pca2 = PCA(2)
    data_pca2 = pca2.fit_transform(npdata_norm)
    tsne2 = TSNE(n_components=2,init='pca',random_state=123)
    data_tsne2 = tsne2.fit_transform(npdata_norm)

    # make a dataframe with labels
    d = {'letter': labels, 'pca1': data_pca2[:,0], 'pca2': data_pca2[:,1]}
    df_pca2 = pd.DataFrame(data=d)

    d = {'letter': labels, 'pca1': data_pca3[:,0], 'pca2': data_pca3[:,1], 'pca3': data_pca3[:,2]}
    df_pca3 = pd.DataFrame(data=d)
    
    d = {'letter': labels, 'tsne1': data_tsne2[:,0], 'tsne2': data_tsne2[:,1]}
    df_tsne2 = pd.DataFrame(data=d)

    # now plot 
    fig = px.scatter(df_pca2, x="pca1", y="pca2", text="letter", size_max=60)
    fig.update_traces(textposition='top center')
    fig.update_layout(height=800,title_text='Letter embedding1')
    plotly.offline.plot(fig,filename='pca2d.html')

    ''' 
    fig = px.scatter(df_tsne2, x="tsne1", y="tsne2", text="letter", size_max=60)
    fig.update_traces(textposition='top center')
    fig.update_layout(height=800,title_text='Letter embedding1')
    plotly.offline.plot(fig,filename='tsne2d.html')
    '''

    # don't do the 3d scatter right now
    '''
    fig = px.scatter_3d(df_pca3, x="pca1", y="pca2", z = "pca3" , text="letter", size_max=60)
    fig.update_traces(textposition='top center')
    fig.update_layout(height=800,title_text='Letter embedding2')
    plotly.offline.plot(fig,filename='plotly_express2.html')
    '''
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    