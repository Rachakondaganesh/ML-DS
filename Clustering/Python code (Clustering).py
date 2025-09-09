## CLUSTERING

### K-means clustering ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=400,n_features=2,centers=3,random_state=1)
plt.scatter(X[:,0],X[:,1],edgecolors="blue")
plt.show() # here you can see 3 clusters



from sklearn.cluster import KMeans
km=KMeans(n_clusters=5,init="random",n_init=10,max_iter=300,tol=0.001,random_state=1)
y_km=km.fit_predict(X)
plt.scatter(X[y_km==0,0],X[y_km==0,1],c="red",label="cluster 1")
plt.scatter(X[y_km==1,0],X[y_km==1,1],c="pink",label="cluster2")
plt.scatter(X[y_km==2,0],X[y_km==2,1],c="green",label="cluster3")
plt.scatter(X[y_km==3,0],X[y_km==3,1],c="orange",label="cluster4")
plt.scatter(X[y_km==4,0],X[y_km==4,1],c="black",label="cluster5")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker="*")
plt.legend(scatterpoints=1)
plt.grid()
plt.show() # here you can see the 5 clusters with centre points in grid format with labels



distortions=[]
max_k=20
for i in range(1,max_k):
    km = KMeans(n_clusters=i, init="random", n_init=10,
                max_iter=300, tol=0.001, random_state=1)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,max_k),distortions)
plt.show() # here you can see the slope line



###Heirarchical clustering:
    # Agglomerative clustering:
    # Divisive clustering:

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import lineStyles

link="C:/Users/user/Downloads/USArrests.csv"
df=pd.read_csv(link)
print(df)
X=df.iloc[:,1:]
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,10))
dend=shc.dendrogram(shc.linkage(X))
plt.title("dendrogram")
plt.axhline(y=27,color="red") # by adding this you can see the red link its upto you weather you add or not
plt.show()











