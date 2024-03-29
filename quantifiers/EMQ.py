import pandas as pd
import numpy as np
from utils.auxiliary import class_dist
import pdb
# it is not working
def EMQ(test_scores, tr_dist, max_it = 1000, eps = 1e-6):
    p_tr = tr_dist
    p_s = np.copy(p_tr)
    p_cond_tr = np.array(test_scores)
    p_cond_s = np.zeros(p_cond_tr.shape)
    
    for it in range(max_it):
        r = p_s / p_tr
        p_cond_s = p_cond_tr * r
        s = np.sum(p_cond_s, axis = 1)
        for c in range(0,len(tr_dist)):
            p_cond_s[:,c] = p_cond_s[:,c] / s
        p_s_old = np.copy(p_s)
        p_s = np.sum(p_cond_s, axis = 0) / p_cond_s.shape[0]
        if (np.sum(np.abs(p_s - p_s_old)) < eps):
            break

    return(p_s/np.sum(p_s))