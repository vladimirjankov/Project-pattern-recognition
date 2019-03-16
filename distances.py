
import math

def hellinger(p, q):
    return sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p,q) ])