
import math
from numpy import dot
from numpy.linalg import norm
def cosine_distance(p,q):
    return  1- dot(p, q)/(norm(p)*norm(q))


def hellinger(p, q):
    return sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p,q) ])



# dodaj jos koju ne budi lenj ... 