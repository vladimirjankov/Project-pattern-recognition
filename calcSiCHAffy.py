from sklearn import metrics
from scipy.spatial.distance import braycurtis,canberra,correlation
from scipy.stats import wasserstein_distance, energy_distance
from support_functions import distance_matrix
from PIL import Image
from distances import hellinger

def calcSiCHAffy(data,label,name):
    silhouette_score_cosine = metrics.silhouette_score(data,label,metric='cosine')
    
    dist_braycurtis = distance_matrix(data,braycurtis)
    silhouette_score_braycurtis = metrics.silhouette_score(dist_braycurtis,label,metric='precomputed')
    
    dist_canberra = distance_matrix(data,canberra)
    silhouette_score_canberra = metrics.silhouette_score(dist_canberra,label,metric='precomputed')
    
    dist_correlation = distance_matrix(data,correlation)
    silhouette_score_correlation = metrics.silhouette_score(dist_correlation,label,metric='precomputed')

    dist_hellinger= distance_matrix(data,hellinger)
    silhouette_score_hellinger = metrics.silhouette_score(dist_hellinger,label,metric='precomputed')

    dist_wasserstein= distance_matrix(data,wasserstein_distance)
    silhouette_score_wasserstein = metrics.silhouette_score(dist_wasserstein,label,metric='precomputed')

    dist_energy_distance= distance_matrix(data,energy_distance)
    silhouette_energy_distance = metrics.silhouette_score(dist_energy_distance,label,metric='precomputed')

    CH = metrics.calinski_harabaz_score(data,label)
    return [name,silhouette_score_cosine,silhouette_score_braycurtis, silhouette_score_canberra,silhouette_score_correlation,
    silhouette_score_hellinger,silhouette_score_wasserstein,silhouette_energy_distance,CH]


 # Rangmatrix,kcz,
 #   img = Image.fromarray(dist_cosine,mode='I')
 #   img.show()
 #   img.save(name+'.png')