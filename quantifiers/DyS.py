import numpy as np
from utils.auxiliary import DyS_distance
from utils.auxiliary import TernarySearch
from utils.auxiliary import getHist

import pdb

#def DyS(pos_scores, neg_scores, test_scores, measure='topose'):
    
#    bin_size = np.linspace(2,10,9)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
#    bin_size = np.append(bin_size, 30)
#    result  = []
    #vDist = []
#    for bins in bin_size:
        #....Creating Histograms bins score\counts for validation and test set...............
        
 #       p_bin_count = qntu.getHist(pos_scores, bins)
 #       n_bin_count = qntu.getHist(neg_scores, bins)
 #       te_bin_count = qntu.getHist(test_scores, bins)
        
 #       def f(x):            
 #           return(qntu.DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
 #       result.append(qntu.TernarySearch(0, 1, f))                                           
                        
  #  pos_prop = round(np.median(result),2)
  #  return pos_prop

def DyS(pos_scores, neg_scores, test_scores, measure='topsoe'):
    bin_size = np.linspace(2,20,10)
    #bin_size = np.linspace(2,10,9)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    bin_size = bin_size.astype(int)

    result  = []
    score_range = (np.min(np.append(pos_scores, neg_scores)), np.max(np.append(pos_scores, neg_scores)))

    for bins in bin_size:
        #....Creating Histograms bins score\counts for validation and test set...............
        
        #p_bin_count = np.histogram(pos_scores, bins=bins, range=score_range)[0]/len(pos_scores)
        #n_bin_count = np.histogram(neg_scores, bins=bins, range=score_range)[0]/len(neg_scores)
        #te_bin_count = np.histogram(test_scores, bins=bins, range=score_range)[0]/len(test_scores)
        p_bin_count = getHist(pos_scores, bins)
        n_bin_count = getHist(neg_scores, bins)
        te_bin_count = getHist(test_scores, bins)
        
        def f(x):            
            return(DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
        result.append(TernarySearch(0, 1, f))                                           
                        
    pos_prop = round(np.median(result),2)
    return pos_prop
    


    
        




