import numpy as np
from quantifiers.FM import FM


def ensembleFM(X_test, list_clf, train_scores):
    y_train = train_scores[0]['label']
    nclasses = len(np.unique(y_train))
    p_hat = np.zeros((len(list_clf), nclasses))
    for m in range(len(list_clf)):
        test_scores = list_clf[m].predict_proba(X_test)
        p_hat[m] = FM(np.array(train_scores[m].iloc[:,:-1]), test_scores, train_scores[m]['label'], nclasses)
    
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))