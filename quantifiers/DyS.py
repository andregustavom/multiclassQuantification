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

def DyS(pos_scores, neg_scores, test_scores, measure='topose'):
    
    bin_size = np.linspace(2,20,10)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    
    alphas = np.zeros(len(bin_size))
    dists = np.zeros(len(bin_size))
    for i, bins in enumerate(bin_size):
        #....Creating Histograms bins score\counts for validation and test set...............
        
        p_bin_count = getHist(pos_scores, bins)
        n_bin_count = getHist(neg_scores, bins)
        te_bin_count = getHist(test_scores, bins)
        
        def f(x):            
            return(DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
        alphas[i] = TernarySearch(0, 1, f)
    
    return np.median(alphas)
    
    

# This DyS uses the scores from a single multiclass classifier
def MultiClassDyS(X_test, classes, list_clf_multi, list_scores_multi):
  
    p_hat = np.zeros((len(list_clf_multi), len(classes)))
    for mi in range(len(list_clf_multi)):

        clf = list_clf_multi[mi]
        all_test_scores = clf.predict_proba(X_test)

        scores = list_scores_multi[mi]    
        
        for c in range(len(classes)):
            pos_scores = scores[scores['label']==c][c]#[x[c] for indx,x in enumerate(train_scores) if train_labels[indx] == classes[c]]
            neg_scores = scores[scores['label']!=c][~c]#[x[c] for indx,x in enumerate(train_scores) if train_labels[indx] != classes[c]]
            test_scores = [x[c] for x in all_test_scores]
            p_hat[mi][c] = DyS(pos_scores, neg_scores, test_scores, measure='topsoe')[0]
    
    p = np.median(p_hat, axis = 0)       
    return(p/np.sum(p))

    



