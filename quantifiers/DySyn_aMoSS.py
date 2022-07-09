import numpy as np
from utils.auxiliary import DyS_distance
from utils.auxiliary import TernarySearch
from utils.auxiliary import getHist
from utils.aMoSS import aMoSS
import pdb


# It's not working! 
def DySyn_aMoSS(pos_scores, neg_scores, test_scores, u_p, u_n, adj_score=False):
     
    MF = [0.1, 0.2, 0.3, 0.4]#, 0.2, 0.4]
    bin_size = np.linspace(2,20,10)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)


    bin_size = bin_size.astype(int)

    result = []
    vdists = []

    score_range = [np.min(np.append(neg_scores, pos_scores)), np.max(np.append(neg_scores, pos_scores))]

    #if np.sum(np.histogram(test_scores, bins=2, range=score_range)[0]/len(test_scores)) <= 0.0:
    #  pdb.set_trace()

    if adj_score == False:
      if (min(test_scores) > 0):
        test_scores = test_scores - min(test_scores)
      else:
        test_scores = test_scores + abs(min(test_scores))

    for bins in bin_size:
      alphas  = []
      dists = []
      for mfi in MF:
        p_scores, n_scores, _ = aMoSS(1000, 0.5, mfi, u_p, u_n)

        #p_bin_count = np.histogram(p_scores, bins=bins, range=score_range)[0]/len(p_scores)
        #n_bin_count = np.histogram(n_scores, bins=bins, range=score_range)[0]/len(n_scores)
        #te_bin_count = np.histogram(test_scores, bins=bins, range=score_range)[0]/len(test_scores)  

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
      vdists.append(min_dist)
    
    pos_prop = np.median(result)
    
    #pos_prop = DyS(pos_scores, neg_scores, test_scores, measure='topose')
    
    
    return pos_prop
