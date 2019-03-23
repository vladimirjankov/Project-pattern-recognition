import numpy as np

def distance_matrix(data,metr):
    m , k = data.shape
    dist_matrix = np.zeros((m,m), dtype=float)

    for i in range(0,m):
        for j in range(i,m):
            dist_matrix[i,j] = metr(data.values[i,:],data.values[j,:])
            dist_matrix[j,i] = dist_matrix[i,j]
    return dist_matrix

def distance_matrix_np(data,metr):
    m , k = data.shape
    dist_matrix = np.zeros((m,m), dtype=float)

    for i in range(0,m):
        for j in range(i,m):
            dist_matrix[i,j] = metr(data[i,:],data[j,:])
            dist_matrix[j,i] = dist_matrix[i,j]
    return dist_matrix

def normalise_data(data):
        data_normed = data
        colum_sum = np.sum(data,axis=0)
        i = 0
        j = 0 
        for i in range(0,data.shape[0]):
                for j in range(0,data.shape[1]):
                        data_normed[i,j]  = data[i,j]/colum_sum[j]
                

        return data_normed        




