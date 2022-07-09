import pandas as pd
import numpy as np
from utils.MoSS import MoSS
from utils.auxiliary import DyS_distance
from utils.auxiliary import TernarySearch
from utils.auxiliary import getHist

import pdb


"""DySyn is a member of the DyS framework that uses synthetic scores to find best match with the test scores distribution.
 
Parameters
----------
test_scores : array
    An array containing the score estimated for the positive class from each test set instance.
Returns
-------
float
    A float value representing the predicted label distribution for positive instances. 
 """
def DySyn(test_scores):
    MF = [0.1,0.3,0.5]
    bin_size = np.linspace(2,20,10)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    bin_size = bin_size.astype(int)

    result = []
    for bins in bin_size:
        alphas = []
        dists = []
        for mfi in MF:
            p_scores, n_scores, _ = MoSS(1000, 0.5, mfi)

            p_bin_count = getHist(p_scores, bins)
            n_bin_count = getHist(n_scores, bins)
            te_bin_count = getHist(test_scores, bins)

            def f(x):            
                return(DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, 'topsoe'))
    
            alpha = TernarySearch(0, 1, f)
            alphas.append(alpha)
            dists.append(DyS_distance(((p_bin_count*alpha) + (n_bin_count*(1-alpha))), te_bin_count, 'topsoe'))

    min_dist = min(dists)
    result.append(alphas[dists.index(min_dist)])
    pos_prop = np.median(result)
    return pos_prop