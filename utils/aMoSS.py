import numpy as np
import pandas as pd
import pdb

def aMoSS_v1(n, alpha, m, u_p, u_n):
    p_scores = ((np.random.uniform(0,1,int(n*alpha)))**m) - (1-u_p)
    n_scores = (1-(np.random.uniform(0,1,int(n*(1- alpha)))**m)) + u_n
    scores  = pd.concat([pd.DataFrame(np.append(p_scores, n_scores)), pd.DataFrame(np.append(['1']*len(p_scores), ['2']*len(n_scores)))], axis=1)
    scores.columns = ['score', 'label']
    return p_scores, n_scores, scores


def aMoSS(n, alpha, m, u_p, u_n):
    p_scores = ((np.random.uniform(0,1,int(n*alpha)))**m) - (1-u_p)
    n_scores = (1-(np.random.uniform(0,1,int(n*(1- alpha)))**m)) + u_n
    scores  = pd.concat([pd.DataFrame(np.append(p_scores, n_scores)), pd.DataFrame(np.append(['1']*len(p_scores), ['0']*len(n_scores)))], axis=1)
    scores.columns = ['score', 'label']
    return p_scores, n_scores, scores