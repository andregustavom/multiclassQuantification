
import numpy as np
import pandas as pd
from utils.aMoSS_ranged import aMoSS_ranged
from utils.auxiliary import DyS_distance
from utils.auxiliary import TernarySearch

# Working on.. TODO
def DyS2Syn(pos_scores, neg_scores, test_scores, u_p, u_n):
     
    MF = [0.2, 0.3, 0.4, 0.6]#, 0.2, 0.4]
    bin_size = np.linspace(2,20,10)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)

    bin_size = bin_size.astype(int)

    result = []

    score_range = [np.min(np.append(neg_scores, pos_scores)), np.max(np.append(neg_scores, pos_scores))]
    for bins in bin_size:
      alphas  = []
      dists = []
      for mfi in MF:
        p_scores, n_scores, _ = aMoSS_ranged(1000, 0.5, mfi, u_p, u_n, score_range)

        p_bin_count = np.histogram(p_scores, bins=bins, range=score_range)[0]/len(p_scores)
        n_bin_count = np.histogram(n_scores, bins=bins, range=score_range)[0]/len(n_scores)
        te_bin_count = np.histogram(test_scores, bins=bins, range=score_range)[0]/len(test_scores)
        
        #CM = np.concatenate([[n_bin_count], [p_bin_count]]).T                  
        #if ~np.isnan(CM).any():
        #  p = cvx.Variable(2)
        #  constraints = [p >= 0, cvx.sum(p) == 1.0]
        #  problem = cvx.Problem(cvx.Minimize(cvx.sum(cvx.kl_div(2 * CM @ p, te_bin_count) +
        #                                            cvx.kl_div(2 * te_bin_count, CM @ p))), constraints)
        #  problem.solve(max_iters=10000)

          #return p.value[0]       
        def f(x):
          return(DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, 'topsoe'))
        
        alpha = TernarySearch(0, 1, f)

        #  if type(p.value) is not type(None):
        #    alpha = np.round(p.value[0], 3)
        alphas.append(alpha)
        dists.append(DyS_distance(((p_bin_count*alpha) + (n_bin_count*(1-alpha))), te_bin_count, 'topsoe'))

      #if len(dists) > 0:
        min_dist = min(dists)
        result.append(alphas[dists.index(min_dist)])
    pos_prop = np.median(result)
    #if len(dists) == 0:
    #  print('DySyn error opt')
     # pos_prop = DyS(pos_scores, neg_scores, test_scores, measure='topose')
    #else:
    #  print('DySyn OK')
    
    return pos_prop
