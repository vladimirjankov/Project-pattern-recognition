from sklearn import metrics,cluster

from scipy.spatial.distance import dice,hamming,jaccard,kulsinski,rogerstanimoto,russellrao,sokalmichener,sokalsneath,yule
from scipy.stats import wasserstein_distance, energy_distance,cosine
from support_functions import distance_matrix,percentage_binary_normalise,distance_matrix_np
from PIL import Image
from distances import hellinger,cosine_distance,dist_kulczynski, jack_knife,dist_kulczynski_vectors,cosine_distance
from matplotlib import pyplot as plt
from clustering_alg import k_means
import numpy as np



def ari_scores_binary_percentage_normed(data,labels,file,p1,p2):
    data = np.transpose(percentage_binary_normalise(data,p1,p2) )
    labels = np.array(labels)
    lbls =np.reshape(labels,(labels.shape[0],))
    ari_dice =0
    ari_hamming =0
    ari_jaccard = 0
    ari_rogerstanimoto =0
    ari_russellrao=0
    ari_sokalmichener=0
    ari_sokalsneath=0
    ari_yule= 0
    for i in range(0,25):
        dice_label = k_means(data, np.max(np.unique(labels)), dice)
        ari_dice = ari_dice+ metrics.adjusted_rand_score(lbls,dice_label)
 #radi normalno

        hamming_label = k_means(data, np.max(np.unique(labels)), hamming)
        ari_hamming =ari_hamming+ metrics.adjusted_rand_score(lbls,hamming_label)
#radi normalno

        jaccard_label = k_means(data, np.max(np.unique(labels)), jaccard)
        ari_jaccard =ari_jaccard+ metrics.adjusted_rand_score(lbls,jaccard_label)
#radi normalno
        rogerstanimoto_label = k_means(data, np.max(np.unique(labels)), rogerstanimoto)
        ari_rogerstanimoto = ari_rogerstanimoto+metrics.adjusted_rand_score(lbls,rogerstanimoto_label)
#radi normalno
        russellrao_label = k_means(data, np.max(np.unique(labels)), russellrao)
        ari_russellrao =ari_russellrao+ari_russellrao+ metrics.adjusted_rand_score(lbls,russellrao_label)
#radi normalno

        sokalmichener_label = k_means(data, np.max(np.unique(labels)), sokalmichener)
        ari_sokalmichener =ari_sokalmichener+ metrics.adjusted_rand_score(lbls,sokalmichener_label)
#radi normalno

        sokalsneath_label = k_means(data, np.max(np.unique(labels)), sokalsneath)
        ari_sokalsneath = ari_sokalsneath + metrics.adjusted_rand_score(lbls,sokalsneath_label)
#radi normalno
        yule_label = k_means(data, np.max(np.unique(labels)), yule)
        ari_yule = ari_yule +metrics.adjusted_rand_score(lbls,yule_label)
#radi normalno
    tmp = np.array([ari_dice,ari_hamming,ari_jaccard,ari_rogerstanimoto,ari_russellrao,ari_russellrao,ari_sokalmichener,ari_sokalsneath,ari_yule])/25

    return [file,tmp.tolist()]

