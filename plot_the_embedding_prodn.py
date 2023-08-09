import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly
import numpy as np



# embeddings come from the makemore code
# which model they come from matters and the depth of the model
# for example if there are more layers after the embedding then
# the meaning of them may vary

# labels = ['0', 'b3', 'f1', 'f3', 'b2', 'f2', 'a114', 'a214', 'a116', 'a216', 's3', 'a224', 'b1', 'b28', 'a125', 'f28', 'a225', 'a126', 'b38', 's2', 'f1*', 'f3*', 'b29', 'f29', 'a115', 'a124', 'b37', 'a215', 'f2d#', 'b27', 'a226', 'f27', 'f1n@', 'f2n#', 'f18', 'f38', 'f3n@', 'b3n@', 'f3w@', 'b1*', 's1', 'b2d@', 'b2d#', 'f1d@', 'b39', 'r2', 'f2d@', 'f1w@', 'b2n#', 'f3d@', 'b3w@', 'f3n#', 'b2n@', 'b1n@', 'b3d@', 'b3n#', 'f1n#', 'f3d#', 'b18', 'b1w@', 'f1w#', 'b3*', 's2n#', 's28', 'f17', 'f2n@', 'f3w#', 'f39', 's38', 'b3w#', 'b1d@', 'f19', 'r3', 'v1*', 's2d#', 'f37', 'r1', 's27', 'f1d#', 's3n#', 'm2', 'r28', 'b3d#', 'r2n#', 'b1w#', 'b1n#', 's29', 'o1*', 'b19', 's37', 'r2d#', 'y1', 'o3*', 'b1d#', 'u3', 'b17', 'z1*', 'r27', 'z3*', 's3n@', 'v3*', 'l2', 'r29', 'f1x@', 's39', 'z1', 'y3', 'z3', 's3w#', 's1w#', 'm3', 's1n#', 'u3*', 'f3x#', 'v3', 'f3x@', 'y1*', 'u1', 'r3n#', 's3d#', 'y1n@', 'f1x#', 'v1', 'r1n#', 'f', 'u1*', 'b', 's2n@', 's1d#', 's3d@', 'z2', 'f3s#', 'm2d#', 'o3', 'u3n@', 'm1', 'b3x@', 'o1', 'r38', 'r18', 'f2*', 'r3d#', 'r3w#', 'f1f#', 'fn#', 'j1*', 'bn#', 'l3', 'b1x@', 's3w@', 'b1x#', 'fd#', 'o2', 'o2*', 'b3x#', 's2d@', 'f1r#', 'y3n@', 's18', 'v2', 'bd#', 'r17', 'r39', 'y3*', 's17', 's1n@', 'l1', 'r37', 'l2d#', 'r1d#', 'u1n@', 'h3', 'r1w#', 'v2*', 'f19w#', 'j3*', 'z2*', 's1*', 's19', 'fw#', 'r19', 's1w@', 'i3', 'f3b#', 'fd@', 'b2*', 'f18w#', 'z1n@', 's1d@', 'f39w#', 'm3w#', 's1x#', 'i2', 'bw#', 'v1n@', 'b38w#', 'i1', 'm3d#', 's3*', 'bn@', 'fn@', 'j1', 'u2', 'b3s#', 'z3n@', 'f17w#', 'f38w#', 'h2', 'b37w#', 'fw@', 'r1*', 'm1w#', 'b1r#', 'ff*', 's3x#', 'l3d#', 'z2n#', 'bf*', 'm3*', 'b19w#', 'b39w#', 'bd@', 'j3', 'f19w@', 'h1', 'sn#', 'v3n@', 'v2n#', 'z2n@', 'b1f#', 'm1d#', 'u2*', 'v3n#', 'y1w@', 'z1n#', 'b19w@', 'l3w#', 'z3n#', 'b18w#', 'b3z#', 'y2', 'z1w#', 'b39w@', 'r3*', 'z1w@', 'r2n@', 'b17w#', 'bw@', 'f18w@', 'sd#', 'r3x#', 't2', 'b38w@', 'o3n@', 'j2', 's', 'r']

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
    kmeans_model = KMeans(n_clusters=8,random_state=0)
    kmeans = kmeans_model.fit(npdata_norm)
    kmeans_clusters = kmeans.predict(npdata_norm)
    # try for lower dim directly
    pca3 = PCA(3)
    data_pca3 = pca3.fit_transform(npdata_norm)
    
    # pca2 with clusters
    pca2 = PCA(2)
    data_pca2 = pca2.fit_transform(npdata_norm)
    cluster_centres_pca = pca2.transform(kmeans.cluster_centers_)
    #cluster_centres_pca
    #cluster_centres_pca.shape[0]
    labels_centres = []
    colours_centres = []
    for i in range(cluster_centres_pca.shape[0]):
       labels_centres.append("Centre Cluster" + str(i))
       colours_centres.append(i)
    
    
    
    tsne2 = TSNE(n_components=2,init='pca',random_state=123,n_iter=1000)
    data_tsne2 = tsne2.fit_transform(npdata_norm)

    # uncomment for kmean cluster
    #labels = kmeans_clusters
    # make a dataframe with labels
    d = {'letter': labels, 'clusters':kmeans_clusters,'pca1': data_pca2[:,0], 'pca2': data_pca2[:,1]}
    df_pca2 = pd.DataFrame(data=d)
    d_centres = {'letter': labels_centres, 'clusters':colours_centres,'pca1': cluster_centres_pca[:,0], 'pca2': cluster_centres_pca[:,1]}

    d = {'letter': labels, 'clusters':kmeans_clusters,'pca1': data_pca3[:,0], 'pca2': data_pca3[:,1], 'pca3': data_pca3[:,2]}
    df_pca3 = pd.DataFrame(data=d)
    
    d = {'letter': labels, 'clusters':kmeans_clusters,'tsne1': data_tsne2[:,0], 'tsne2': data_tsne2[:,1]}
    df_tsne2 = pd.DataFrame(data=d)

    # now plot 
    #fig = px.scatter(df_pca2, x="pca1", y="pca2", text="letter", color="clusters",size_max=60)
    #fig.update_traces(textposition='top center')
    #fig.update_layout(height=800,title_text='Letter embedding1')
    #plotly.offline.plot(fig,filename='pca2d.html')
    #?fig.add_scatter
    fig = px.scatter(df_pca2, x="pca1", y="pca2", text="letter", color="clusters",size_max=60)
    scatter2 = px.scatter(d_centres, x="pca1", y="pca2", text="letter", color="clusters")
    scatter2.update_traces(marker=dict(symbol='cross',size=15))
    fig.add_trace(scatter2.data[0])
    fig.update_layout(height=800,title_text='Letter embedding1')
    plotly.offline.plot(fig,filename='pca2.html')

     
    fig = px.scatter(df_tsne2, x="tsne1", y="tsne2", text="letter", color="clusters", size_max=60)
    fig.update_traces(textposition='top center')
    fig.update_layout(height=800,title_text='Letter embedding1')
    plotly.offline.plot(fig,filename='tsne2d.html')
    

    # don't do the 3d scatter right now
    '''
    fig = px.scatter_3d(df_pca3, x="pca1", y="pca2", z = "pca3" , text="letter", size_max=60)
    fig.update_traces(textposition='top center')
    fig.update_layout(height=800,title_text='Letter embedding2')
    plotly.offline.plot(fig,filename='plotly_express2.html')
    '''
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    