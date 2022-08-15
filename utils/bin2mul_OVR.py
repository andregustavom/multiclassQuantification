from utils.quantifierTemplate import apply_quantifier
from utils.auxiliary import getTPRandFPRbyThreshold
import numpy as np
from copy import deepcopy


def binary2multi_OVR(X_test, counter, list_scorer_bin, list_scores_bin, n_classes):
    vdist = ["topsoe", "jensen_difference", "prob_symm", "ord", "sord", "hellinger"] 
    names_vdist = ["TS", "JD", "PS", "ORD", "SORD", "HD"] 
    allow_proba = True
     # Running our methods
    aux = counter.split("-")
    quantifier = counter
    measure = 'topsoe'
    # Selecting the measure (topsoe, hellinger, etc)
    if len(aux) > 1:
        quantifier = aux[0]
        measure = vdist[names_vdist.index(aux[1])]
    # Performeing OVR scheme
    
    dist_predicted = []
    for ci in range(n_classes):
        clf = list_scorer_bin[ci]
        scores = deepcopy(list_scores_bin[ci])                     

        # Classifiers that provides the probabilities of each sample
        if allow_proba:
            te_scores = clf.predict_proba(X_test)[:,1]  #estimating test sample scores                
        else:
            # SVC()
            te_scores = clf.decision_function(X_test)  #estimating test sample scores 
        if 'aMoSS' in quantifier:
            if (min(scores.score) > 0):  
                scores.score = scores.score - min(scores.score)
            else:
                scores.score = scores.score + abs(min(scores.score))                      

        tprfpr = getTPRandFPRbyThreshold(scores)           
        pos_scores = scores[scores['label']==1]['score']
        neg_scores = scores[scores['label']==0]['score']
        u_p = np.mean(pos_scores)
        u_n = np.mean(neg_scores)

        #.............Calling of Methods..................................................  
        pred_pos_prop = apply_quantifier(qntMethod = quantifier, p_score = pos_scores, n_score = neg_scores, test_score = te_scores, 
                                TprFpr = tprfpr, thr = np.round(np.median(tprfpr['threshold']),1), measure = measure, calib_clf = None, 
                                X_test = X_test, u_p = u_p, u_n = u_n, adj_score=allow_proba)  

        if np.isscalar(pred_pos_prop) is False:
            pred_pos_prop = pred_pos_prop[0]

        pred_pos_prop = np.round(pred_pos_prop,2)  #predicted class proportion
        dist_predicted.append(pred_pos_prop)
    return dist_predicted
                     