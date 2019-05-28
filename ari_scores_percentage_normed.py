from sklearn import metrics,cluster

from scipy.spatial.distance import braycurtis,canberra,correlation,euclidean
from scipy.stats import wasserstein_distance, energy_distance,cosine
from support_functions import distance_matrix,percentage_normalise,distance_matrix_np
from PIL import Image
from distances import hellinger,cosine_distance,dist_kulczynski, jack_knife,dist_kulczynski_vectors,cosine_distance
from matplotlib import pyplot as plt
from clustering_alg import k_means
import numpy as np



def ari_scores_percentage_normed(data,labels,file,p1,p2):
    data = np.transpose(percentage_normalise(data,p1,p2) )
    labels = np.array(labels)
    lbls =np.reshape(labels,(labels.shape[0],))
    ari_cosine =0 
    ari_braycurtis =0
    ari_correlation=0
    ari_canberra=0
    ari_hellinger=0
    ari_wasserstein=0
    ari_energy=0
    ari_kulczynski=0
    ari_eucl=0
    for i in range(0,25):
        cosine_label = k_means(data, np.max(np.unique(labels)),cosine_distance)
        ari_cosine = ari_cosine+metrics.adjusted_rand_score(lbls,cosine_label)

        braycurtis_label = k_means(data, np.max(np.unique(labels)), braycurtis)
        ari_braycurtis = ari_braycurtis+metrics.adjusted_rand_score(lbls,braycurtis_label)
 #radi normalno

        correlation_label = k_means(data, np.max(np.unique(labels)), correlation)
        ari_correlation =ari_correlation+ metrics.adjusted_rand_score(lbls,correlation_label)
#radi normalno

        canberra_label = k_means(data, np.max(np.unique(labels)), canberra)
        ari_canberra = ari_canberra+ metrics.adjusted_rand_score(lbls,canberra_label)
#radi normalno
        hellinger_label = k_means(data, np.max(np.unique(labels)), hellinger)
        ari_hellinger =ari_hellinger+ metrics.adjusted_rand_score(lbls,hellinger_label)
#radi normalno
        wasserstein_distance_label = k_means(data, np.max(np.unique(labels)), wasserstein_distance)
        ari_wasserstein = ari_wasserstein+ metrics.adjusted_rand_score(lbls,wasserstein_distance_label)
#radi normalno

        energy_distance_label = k_means(data, np.max(np.unique(labels)), energy_distance)
        ari_energy = ari_energy + metrics.adjusted_rand_score(lbls,energy_distance_label)
#radi normalno

        kulczynski_label = k_means(data, np.max(np.unique(labels)), dist_kulczynski_vectors)
        ari_kulczynski = ari_kulczynski+metrics.adjusted_rand_score(lbls,kulczynski_label)
#radi normalno
        eucl_label = k_means(data, np.max(np.unique(labels)), euclidean)
        ari_eucl = ari_eucl + metrics.adjusted_rand_score(lbls,eucl_label)
#radi normalno
    tmp = np.array([ari_cosine,ari_braycurtis,ari_correlation,ari_canberra,ari_hellinger,ari_wasserstein,ari_energy,ari_kulczynski,ari_eucl]) / 25

    return [file, tmp.tolist()]



