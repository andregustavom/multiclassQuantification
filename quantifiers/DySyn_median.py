import numpy as np
from quantifiers.DyS import DyS
from utils.MoSS import MoSS

def DySyn_median(test_scores):
     
    MF = [0.1,0.3,0.5]
    result  = []
    for mfi in MF:        
        p_scores, n_scores, _ = MoSS(1000, 0.5, mfi)
        prop = DyS(p_scores, n_scores, test_scores, 'topsoe')
        result.append(prop)                                           
                        
    pos_prop = np.median(result)
    return pos_prop
