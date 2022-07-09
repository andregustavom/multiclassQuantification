

import numpy as np
import pandas as pd
import cvxpy as cvx

def DyS_opt(pos_scores, neg_scores, test_scores, measure='topsoe'):
    
    bin_size = np.linspace(2,20,10)
    bin_size = np.append(bin_size, 30)
   
    bin_size = bin_size.astype(int)
    
    result  = []
    score_range = (np.min(np.append(pos_scores, neg_scores)), np.max(np.append(pos_scores, neg_scores)))
    for bins in bin_size:
        #....Creating Histograms bins score\counts for validation and test set...............        
        p_bin_count = np.histogram(pos_scores, bins=bins, range=score_range)[0]/len(pos_scores)
        n_bin_count = np.histogram(neg_scores, bins=bins, range=score_range)[0]/len(neg_scores)
        te_bin_count = np.histogram(test_scores, bins=bins, range=score_range)[0]/len(test_scores)
        
        CM = np.concatenate([[n_bin_count], [p_bin_count]]).T        
        p = cvx.Variable(2)
        constraints = [p >= 0, cvx.sum(p) == 1.0]
        problem = cvx.Problem(cvx.Minimize(cvx.sum(cvx.kl_div(2 * CM @ p, te_bin_count) +
                                                   cvx.kl_div(2 * te_bin_count, CM @ p))),
                              constraints)
        problem.solve(max_iters=10000)
        #return p.value[0]
        result.append(p.value[0])
        
                        
    pos_prop = round(np.median(result),2)
    return pos_prop