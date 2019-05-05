import numpy as np
import pandas as pd
import os
from calcSiCHAffy import calcSiCHAffy
from matplotlib import pyplot as plt
from ari_scores_normed import ari_scores_normed
import plotly.graph_objs as go
from support_functions import rgb_color_random
import xlsxwriter
import plotly.io as pio




path_data = '/home/vladimir/Desktop/po_projekat/Affy/data/'
path_label = '/home/vladimir/Desktop/po_projekat/Affy/labels/'
file_header = (["   ", "Silhouette_cosine","Silhouette_braycurtis","Silhouette_canberra", "Silhouette_pearson", "Silhouette_hellinger","Silhouette_wasserstein",
    "Silhouette_energy","Silhouette_kulczynski","Calinski_harabaz","Davies_Bouldin"])

distance_names=['cosine', 'braycurtis', 'pearson','canberra','hellinger','wasserstein','energy','kulczynski','eucl']

data_file_names =  sorted(os.listdir(path_data))
label_file_names =  sorted(os.listdir(path_label))

scores = []
data_book = xlsxwriter.Workbook("/home/vladimir/Desktop/po_projekat/Affy/ARI/ARI_normed_results.xlsx")
worksheet = data_book.add_worksheet()


for i in range(0,len(data_file_names)):
    path_data_tmp = path_data + data_file_names[i]
    data = pd.read_csv(path_data_tmp, delimiter=' ')
    label = pd.read_csv(path_label+ label_file_names[i], delimiter=' ')
    ari = ari_scores_normed(np.array(data,dtype=float),label,data_file_names[i])
    scores.append(ari)
print('\n')
print(scores)

row =1
col =0

for item in file_header :
    worksheet.write(0,col,item)
    col = col +1

col = 0

for file_scores in scores:
    for item in file_scores:
        worksheet.write(row,col,item)
        col = col +1
    row = row +1
    col =0

data_book.close()
scrs = [scores[i][1:10] for i in range(0,21)]
scrs = np.array(scrs)

trace = []
for i in range(0,len(distance_names)):
    trace1 = go.Bar(
        x=data_file_names,
        y= scrs[:,i],
        name=distance_names[i],
        marker=dict(
            color=rgb_color_random()
        )
    )
    trace.append(trace1)
    print(scrs[:,i])

data = trace
layout = go.Layout(
   # xaxis=dict(tickangle=-45),
    barmode='group',
)

fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, file='/home/vladimir/Desktop/po_projekat/Affy/ARI/ari_normed_scores.png', format='png')





#np.savetxt('calculations.txt', scores,fmt='%s')

    #         cosine             braycurtis               canberra                person            hellinger         wasserstein       energy distance    kcz jackknife   calinski harabaz
   #         0.6706 je najveci koeficijent za sill i dobijen je sa wasserstein_distancom...
#jackknife  uradjen sa wasserstein_distancom umesto pearsonovom ..

# dodaj euklidsko
# ARI sa kmeans predikcijom
# sa novim normalizacijama
# bez novih normalizacija
# uradi u zasebnom fajlu nove normalizacije i ARI

# proveri funkcije dal rade... na prvi pogled sve deluje okej 


