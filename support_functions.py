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
        
        i = 0
        j = 0 
        for i in range(0,data.shape[0]):
                colum_sum = np.sum(data[i,:])
                for j in range(0,data.shape[1]):                       
                        data_normed[i,j]  = data[i,j]/colum_sum
                
        return data_normed

def percentage_normalise(data,p1,p2):
        # p1 treshold ispod kog je gen ne bitan
        # p2 treshold zastupljenosti gena u svim uzorcima 

        data = normalise_data(data)
        data_smaller = np.array([])
        counter = 0
        for i in range(0,data.shape[1]):
                data[:,i] = data[:,i]/max(data[:,i])
                for j in range(0, data.shape[0]):
                        if data[j,i] < p1:
                                data[j,i] = 0
                                counter = counter + 1
                
                if (counter/float(data.shape[0])) >p2 : 
                        data_smaller.append(data[:,i])
        return data_smaller

def percentage_binary_normalise(data,p1,p2):
        # p1 treshold ispod kog je gen ne bitan
        # p2 treshold zastupljenosti gena u svim uzorcima 

        data = normalise_data(data)
        data_smaller = np.array([])
        counter = 0
        for i in range(0,data.shape[1]):
                data[:,i] = data[:,i]/max(data[:,i])
                data[:,i] = data[:,i] <p1
                counter = np.sum(data[:,i])
                if (counter/float(data.shape[0])) >p2 : 
                        data_smaller.append(data[:,i])
                        
        return data_smaller
                        





