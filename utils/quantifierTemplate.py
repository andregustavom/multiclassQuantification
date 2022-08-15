
from quantifiers.ACC import ACC
from quantifiers.PCC import PCC
from quantifiers.PACC import PACC
from quantifiers.HDy import HDy
from quantifiers.X import X
from quantifiers.MAX import MAX
from quantifiers.SMM import SMM  
from quantifiers.DyS import DyS
from quantifiers.SORD import SORD
from quantifiers.MS import MS
from quantifiers.T50 import T50
from quantifiers.EMQ import EMQ
from quantifiers.CC import CC
from quantifiers.DySyn import DySyn
from quantifiers.DySyn_aMoSS import DySyn_aMoSS
from quantifiers.DyS_opt import DyS_opt
from quantifiers.DySyn_median import DySyn_median
from quantifiers.DySyn_dev import DySyn_aMoSS_D


import pandas as pd
import numpy as np

import pdb
"""This function is an interface for running different quantification methods.
 
Parameters
----------
qntMethod : string
    Quantification method name
p_score : array
    A numeric vector of positive scores estimated either from a validation set or from a cross-validation method.
n_score : array
    A numeric vector of negative scores estimated either from a validation set or from a cross-validation method.
test : array
    A numeric vector of scores predicted from the test set.
TprFpr : matrix
    A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
thr : float
    The threshold value for classifying and counting. Default is 0.5.
measure : string
    Dissimilarity function name used by the DyS method. Default is "topsoe".

Returns
-------
array
    the class distribution of the test calculated according to the qntMethod quantifier. 
 """
def apply_quantifier(qntMethod, p_score, n_score, test_score, TprFpr, thr, measure, calib_clf, X_test, u_p, u_n, adj_score=False):
  if qntMethod == "CC":
    return CC(test_score, thr)
  if qntMethod == "ACC":        
    return ACC(test_score, TprFpr)
  if qntMethod == "EMQ":
    tr_dist = [len(p_score), len(n_score)]
    tr_dist = np.round(tr_dist/np.sum(tr_dist),4)
    test_score = pd.concat([pd.DataFrame(test_score), pd.DataFrame(1-test_score)], axis=1)
    #test_score.columns = ['1', '0']
    return EMQ(np.array(test_score), tr_dist)
  if qntMethod == "SMM":
    return SMM(p_score, n_score, test_score)
  if qntMethod == "HDy":
    return HDy(p_score, n_score, test_score)
  if qntMethod == "DyS+opt":
    return DyS_opt(p_score, n_score, test_score, measure)
  if qntMethod == "DyS":
    return DyS(p_score, n_score, test_score, measure)
  if qntMethod == "DySyn":
    return DySyn(test_score)
    
  if qntMethod == "DySyn+median":
    return DySyn_median(test_score)

  if qntMethod == "DySyn+aMoSS":
    return DySyn_aMoSS(p_score, n_score, test_score, u_p, u_n, adj_score)

  if qntMethod == "DySyn+aMoSS+D":
    return DySyn_aMoSS_D(p_score, n_score, test_score, u_p, u_n)

  if qntMethod == "SORD":
    return SORD(p_score, n_score, test_score)

  if qntMethod == "MS":
    return MS(test_score, TprFpr)
  if qntMethod == "MAX":
    return MAX(test_score, TprFpr)
  if qntMethod == "X":
    return X(test_score, TprFpr)
  if qntMethod == "T50":
    return T50(test_score, TprFpr)
  if qntMethod == "PCC":
    return PCC(calib_clf, X_test,thr)
  if qntMethod == "PACC":
    return PACC(calib_clf, X_test, TprFpr, thr)
  print('ERROR - '+ qntMethod + ' was not found!')
  return None


from quantifiers.ensembleEMQ import ensembleEM
from quantifiers.ensembleFM import ensembleFM
from quantifiers.ensembleGAC import ensembleGAC
from quantifiers.ensembleGPAC import ensembleGPAC



def apply_ensemble_quantifier(qntMethod, X_test, list_clf, list_scores, y_train):

  if qntMethod == 'e_EMQ':
    return ensembleEM(X_test, list_clf, y_train)
  if qntMethod == 'e_FM':
    return ensembleFM(X_test, list_clf, list_scores)
  if qntMethod == 'e_GAC':
    return ensembleGAC(X_test, list_clf, list_scores)
  if qntMethod == 'e_GPAC':
    return ensembleGPAC(X_test, list_clf, list_scores)
