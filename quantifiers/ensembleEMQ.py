from quantifiers.EMQ import EMQ
import numpy as np
from utils.auxiliary import class_dist

def ensembleEM(X_test, list_clf, y_train):
    nclasses = len(np.unique(y_train))
    p_hat = np.zeros((len(list_clf), nclasses))    
    for m in range(len(list_clf)):
        test_scores = list_clf[m].predict_proba(X_test)
        p_hat[m] = EMQ(test_scores, class_dist(y_train, nclasses))
    
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))