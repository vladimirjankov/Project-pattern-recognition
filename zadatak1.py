import numpy as np
import pandas as pd
import os
from calcSiCHAffy import calcSiCHAffy

path_data = '/home/vladimir/Desktop/po_projekat/Affy/data/'
path_label = '/home/vladimir/Desktop/po_projekat/Affy/labels/'

data_file_names =  sorted(os.listdir(path_data))
label_file_names =  sorted(os.listdir(path_label))

scores = []

for i in range(0,len(data_file_names)):
    path_data_tmp = path_data + data_file_names[i]
    data = pd.read_csv(path_data_tmp, delimiter=' ')
    label = pd.read_csv(path_label+ label_file_names[i], delimiter=' ')
    label = np.array(label)
    scores.append(calcSiCHAffy(data,np.ravel(label),data_file_names[i]))
print('\n')
print(scores)

np.savetxt('calculations.txt', scores,fmt='%s')


#napravi slike od matrica
#jackknife jeste da daje shit rez ali trazeno je ... napravi.
#RM isto...
#

# samo cepaj
#                cosine             braycurtis               canberra                person            hellinger         wasserstein       energy distance        calinski harabaz
            #0.6706 je najveci koeficijent za sill i dobijen je sa wasserstein_distancom...


