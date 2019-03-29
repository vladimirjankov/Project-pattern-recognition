from sklearn import metrics
from scipy.spatial.distance import braycurtis,canberra,correlation
from scipy.stats import wasserstein_distance, energy_distance,cosine
from support_functions import distance_matrix,normalise_data,distance_matrix_np
from PIL import Image
from distances import hellinger,cosine_distance,dist_kulczynski, jack_knife
from matplotlib import pyplot as plt
import numpy as np

def calcSiCHAffy(data,label,name):

    normed_data = normalise_data(np.array(data,dtype=float))

    dist_cosine = distance_matrix(data,cosine_distance)
    silhouette_score_cosine = metrics.silhouette_score(dist_cosine,label,metric='precomputed')
    plt.imshow(dist_cosine,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_cosine.jpg')
    plt.close()


    dist_braycurtis = distance_matrix(data,braycurtis)
    silhouette_score_braycurtis = metrics.silhouette_score(dist_braycurtis,label,metric='precomputed')
    plt.imshow(dist_braycurtis,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_braycurtis.jpg')
    plt.close()

    
    dist_canberra = distance_matrix(data,canberra)
    silhouette_score_canberra = metrics.silhouette_score(dist_canberra,label,metric='precomputed')
    plt.imshow(dist_canberra,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_canberra.jpg')
    plt.close()

    
    dist_correlation = distance_matrix(data,correlation)
    silhouette_score_correlation = metrics.silhouette_score(dist_correlation,label,metric='precomputed')
    plt.imshow(dist_correlation,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_pearson.jpg')
    plt.close()
    

    dist_hellinger= distance_matrix_np(normed_data,hellinger)
    silhouette_score_hellinger = metrics.silhouette_score(dist_hellinger,label,metric='precomputed')
    plt.imshow(dist_hellinger,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_hellinger.jpg')
    plt.close()



    dist_wasserstein= distance_matrix_np(normed_data,wasserstein_distance)
    silhouette_score_wasserstein = metrics.silhouette_score(dist_wasserstein,label,metric='precomputed')
    plt.imshow(dist_wasserstein,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_wasserstein.jpg')
    plt.close()


    dist_energy_distance= distance_matrix(data,energy_distance)
    silhouette_energy_distance = metrics.silhouette_score(dist_energy_distance,label,metric='precomputed')
    plt.imshow(dist_energy_distance,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_energy_distance.jpg')
    plt.close()

    dist_kulczyn= dist_kulczynski(np.array(data),strict=True)
    silhouette_kulczynski = metrics.silhouette_score(dist_kulczyn,label,metric='precomputed')
    plt.imshow(dist_kulczyn,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_kulczynski.jpg')
    plt.close()

    dist_eucl= distance_matrix(data,np.linalg.norm)
    silhouette_eucl = metrics.silhouette_score(dist_eucl,label,metric='precomputed')
    plt.imshow(dist_eucl,cmap='autumn')
    plt.colorbar()
    plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_eucl.jpg')
    plt.close()


   # dist_jk = distance_matrix(data,jack_knife)
  #  silhouette_jk = metrics.silhouette_score(dist_jk,label,metric='precomputed')
  #  plt.imshow(dist_jk,cmap='autumn')
  #  plt.colorbar()
 #  plt.savefig('Affy/distance_map/'+name[0:len(name)-4] +'_jackknife.jpg')
 #   plt.close() # ,silhouette_jk ubaci kad budes racunao i njega. spor je


    CH = metrics.calinski_harabaz_score(data,label)
    DB = metrics.davies_bouldin_score(data,label)
    return [name,silhouette_score_cosine,silhouette_score_braycurtis, silhouette_score_canberra,silhouette_score_correlation,
    silhouette_score_hellinger,silhouette_score_wasserstein,silhouette_energy_distance,silhouette_kulczynski,silhouette_eucl,CH,DB]
    
