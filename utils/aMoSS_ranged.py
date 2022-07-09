import numpy as np
import pandas as pd



def aMoSS_ranged(n, alpha, m, u_p, u_n, range_sc):

    #p_scores = ((np.random.uniform(range_sc[0],range_sc[1],int(n*alpha)))**m) - (1-u_p)   

    p_scores = ((np.random.uniform(range_sc[0],range_sc[1],int(n*alpha)))**m) 
    #aux = np.random.uniform(range_sc[0],range_sc[1],int(n*alpha))
    #p_scores = np.array([ (float(x)**m).real for x in aux ])
    #p_scores = p_scores - (1-u_p)

    #n_scores = (1-(np.random.uniform(range_sc[0],range_sc[1],int(n*(1- alpha)))**m)) + u_n

    n_scores = ((np.random.uniform(range_sc[0],range_sc[1],int(n*(1- alpha)))**m)) 
    #aux_n = np.random.uniform(range_sc[0],range_sc[1],int(n*(1- alpha)))
    #n_scores = np.array([ (float(x)**m).real for x in aux_n ])
    #n_scores = (1 - n_scores) + u_n
    
    #p_scores = p_scores[~np.isnan(p_scores)]
    #n_scores = n_scores[~np.isnan(n_scores)]
    scores  = pd.concat([pd.DataFrame(np.append(p_scores, n_scores)), pd.DataFrame(np.append(['1']*len(p_scores), ['0']*len(n_scores)))], axis=1)
    scores.columns = ['score', 'label']
    
    return p_scores, n_scores, scores