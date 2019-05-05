import numpy as np
import random

def init_centroids(data,k):
    i = list(range(0,data.shape[0]))
    random.shuffle(i)
    centroids  = []
    for p in range(0,k):
        centroids.append(data[i[p]])
    return np.array(centroids,dtype=float)

def cluster_init(data,k,centorids,distance):
    clusters = []
    centorids = np.array(centorids,dtype=float)
    for i in range(0,data.shape[0]):
        min_distance = 500000000.0
        k=0
        
        for j in range(0,centorids.shape[0]):
            dist = distance(data[i,:],centorids[j,:])
            if(min_distance > dist) :
                k =j+1
                min_distance = dist
               
        clusters.append(k)        

    return np.array(clusters, dtype=int)

def new_centroid_cal(data,clusters,k):
    cenroids = []
    for i in range(1,k+1):
        cenroid = np.zeros(data[0,:].shape)
        counter = 0
        for j in range(0,data.shape[0]):
            if clusters[j] == i:
                cenroid = cenroid + data[j,:]
                counter = counter +1
        centTemp = np.zeros(cenroid.shape)
        if counter > 0:
            centTemp = cenroid/counter

        cenroids.append(centTemp)
    return np.array(cenroids,dtype=float)
## greska mala izmeni nakon tusiranja




def k_means(data, k, dist_func):
    centroids = init_centroids(data,k)
    clusters = cluster_init(data,k,centroids,dist_func)
    stop = True
    while(stop):
        new_centroid = new_centroid_cal(data,clusters,k)
        new_clusters = cluster_init(data,k,new_centroid,dist_func)
        num_diff = np.sum(clusters ==new_clusters,axis=0)
        if num_diff == clusters.shape[0]:
            stop = False
        return new_clusters



#RuntimeWarning: invalid value encountered in true_divide
 # cenroids.append(cenroid/counter)
 #nesto se ovde cudno desava... 
 #debaguje lagano 





