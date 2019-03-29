from sklearn import metrics,cluster
from scipy.spatial.distance import braycurtis,canberra,correlation
from scipy.stats import wasserstein_distance, energy_distance,cosine
from support_functions import distance_matrix,normalise_data,distance_matrix_np
from PIL import Image
from distances import hellinger,cosine_distance,dist_kulczynski, jack_knife
from matplotlib import pyplot as plt
from clustering_alg import k_means
import numpy as np



def ari_scores_normed(data,labels):
    data = normalise_data(data) 
    labels = np.array(labels)
 #   dist_cosine = distance_matrix(data,cosine_distance)
 #   dist_braycurtis = distance_matrix(data,braycurtis)   
  #  dist_canberra = distance_matrix(data,canberra)
  #  dist_correlation = distance_matrix(data,correlation)
  #  dist_hellinger= distance_matrix_np(normed_data,hellinger)
  #  dist_wasserstein= distance_matrix_np(normed_data,wasserstein_distance)
  #  dist_energy_distance= distance_matrix(data,energy_distance)
   # dist_kulczyn= dist_kulczynski(np.array(data),strict=True)
   # dist_eucl= distance_matrix(data,eucl_distance)

  #  cosine_labels = iterate_k_means(data,,300,cosine_distance)
   # labels = np.transpose(labels)
    cosine_label = k_means(data, np.max(np.unique(labels)), correlation)
    ari_cosine = metrics.adjusted_rand_score(np.reshape(labels,(labels.shape[0],)),cosine_label)

    return ari_cosine

