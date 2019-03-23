
import math
from numpy import dot,isfinite,zeros,shape,rank,where
from numpy.linalg import norm
from astropy.stats import jackknife_resampling
from scipy.stats import wasserstein_distance

def cosine_distance(p,q):
    return  1- dot(p, q)/(norm(p)*norm(q))


def hellinger(p, q):
    return sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p,q) ])


def dist_kulczynski(datamtx, strict=True):
    """ calculates the kulczynski distances between rows of a matrix
    
    see for example Faith et al., composiitonal dissimilarity, 1987
    returns a distance of 1 between a row of zeros and a row with at least one
    nonzero element
    
    * comparisons are between rows (samples)
    * input: 2D numpy array.  Limited support for non-2D arrays if 
    strict==False
    * output: numpy 2D array float ('d') type.  shape (inputrows, inputrows)
    for sane input data
    * two rows of all zeros returns 0 distance between them
    * an all zero row compared with a not all zero row returns a distance of 1
    * if strict==True, raises ValueError if any of the input data is negative,
    not finite, or if the input data is not a rank 2 array (a matrix).
    * if strict==False, assumes input data is a matrix with nonnegative 
    entries.  If rank of input data is < 2, returns an empty 2d array (shape:
    (0, 0) ).  If 0 rows or 0 colunms, also returns an empty 2d array.
    """
    if strict:
        if not isfinite(datamtx).all:
            raise ValueError("non finite number in input matrix")
#        if (datamtx.any<0.0):
#            raise ValueError("negative value in input matrix")
        if rank(datamtx) != 2:
            raise ValueError("input matrix not 2D")
        numrows, numcols = shape(datamtx)
    else:
        try:
            numrows, numcols = shape(datamtx)
        except ValueError:
            return zeros((0,0),'d')

    if numrows == 0 or numcols == 0:
        return zeros((0,0),'d')
    dists = zeros((numrows,numrows),'d')
    rowsums = datamtx.sum(axis=1)
    # rowsum: the sum of elements in a row
    # cache to avoid recalculating for each pair
    for i in range(numrows):
        irowsum = rowsums[i]
        r1 = datamtx[i]
        for j in range(i):
            r2 = datamtx[j]
            jrowsum = rowsums[j]
            rowminsum = float(sum(where(r1<r2, r1,r2)))
            if (irowsum == 0.0 and jrowsum == 0.0):
                cur_d = 0.0 # => two rows of zeros
            elif (irowsum == 0.0 or jrowsum == 0.0):
                cur_d = 1.0 # one row zeros, one not all zeros
            else:
                cur_d = 1.0 - (((rowminsum/irowsum) + (rowminsum/jrowsum))/2.0)
            dists[i][j] = dists[j][i] = cur_d
    return dists



def jack_knife(p,q):
    p_jk_resampling = jackknife_resampling(p)
    q_jk_resampling = jackknife_resampling(q)
    min_dist = wasserstein_distance(p_jk_resampling[0,:],q_jk_resampling[0,:])
    m,k = p_jk_resampling.shape
    for i in range(1,m):
        temp_dist = wasserstein_distance(p_jk_resampling[i,:],q_jk_resampling[i,:])
        if min_dist > temp_dist:
            min_dist = temp_dist
        
    return min_dist



# dodaj jos koju ne budi lenj ... 