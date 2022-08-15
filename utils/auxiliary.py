import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import importlib
from collections import Counter
from sklearn import preprocessing

from os import listdir
from os.path import isfile, join

import os.path
import pickle


from sklearn import svm

from copy import deepcopy

from sklearn.linear_model import RidgeClassifier

import helpers


import pdb

#------------------------------------------------------------------------------------------------------------------------------
class Distances(object):
    
    def __init__(self,P,Q):
        if sum(P)<1e-20 or sum(Q)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P)!=len(Q):
            raise "Arrays need to be of equal sizes..."
        #use numpy arrays for efficient coding
        P=np.array(P,dtype=float);Q=np.array(Q,dtype=float)
        #Correct for zero values
        P[np.where(P<1e-20)]=1e-20
        Q[np.where(Q<1e-20)]=1e-20
        self.P=P
        self.Q=Q
        
    def sqEuclidean(self):
        P=self.P; Q=self.Q; d=len(P)
        return sum((P-Q)**2)
    
    def probsymm(self):
        P=self.P; Q=self.Q; d=len(P)
        return 2*sum((P-Q)**2/(P+Q))
    
    def topsoe(self):
        P=self.P; Q=self.Q
        return sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))
    def hellinger(self):
        P=self.P; Q=self.Q
        return 2 * np.sqrt(1 - sum(np.sqrt(P * Q)))

#------------------------------------------------------------------------------------------------------------------------------
def DyS_distance(sc_1, sc_2, measure='topsoe'):
    
    dist = Distances(sc_1, sc_2)

    if measure == 'sqEuclidean':
        return dist.sqEuclidean()
    if measure == 'topsoe':
        return dist.topsoe()
    if measure == 'probsymm':
        return dist.probsymm()
    if measure == 'hellinger':
        return dist.hellinger()
    print("Error, unknown distance specified, returning topsoe")
    return dist.topsoe()

#------------------------------------------------------------------------------------------------------------------------------
def TernarySearch(left, right, f, eps=1e-4):

    while True:
        if abs(left - right) < eps:
            return(left + right) / 2
    
        leftThird  = left + (right - left) / 3
        rightThird = right - (right - left) / 3
    
        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird 

#------------------------------------------------------------------------------------------------------------------------------
def getHist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins)+1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks,1.1)
    
    re = np.repeat(1/(len(breaks)-1), (len(breaks)-1))  
    for i in range(1,len(breaks)):
        re[i-1] = (re[i-1] + len(np.where((scores >= breaks[i-1]) & (scores < breaks[i]))[0]) ) / (len(scores)+1)
    return re

#------------------------------------------------------------------------------------------------------------------------------
def class_dist(Y, nclasses):
    return np.array([np.count_nonzero(Y == i) for i in range(nclasses)]) / Y.shape[0]


#------------------------------------------------------------------------------------------------------------------------------
def class2index(labels, classes):
    return np.array([classes.index(labels[i]) for i in range(labels.shape[0])])
#------------------------------------------------------------------------------------------------------------------------------
# Requires some adjustments
def getTPRandFPRbyThreshold (validation_scores):
    unique_scores = np.arange(0,1,0.01)
    #unique_scores = np.linspace(min(validation_scores.score), max(validation_scores.score), 100)
    arrayOfTPRandFPRByTr = pd.DataFrame(columns=['threshold','fpr', 'tpr'])
    total_positive = len(validation_scores[validation_scores['label']==1])
    total_negative = len(validation_scores[validation_scores['label']==0])
    for threshold in unique_scores:
        fp = len(validation_scores[(validation_scores['score'] > threshold) & (validation_scores['label']==0)])
        tp = len(validation_scores[(validation_scores['score'] > threshold) & (validation_scores['label']==1)])
        tpr = round(tp/total_positive,2)
        fpr = round(fp/total_negative,2)
    
        aux = pd.DataFrame([[round(threshold,2), fpr, tpr]])
        aux.columns = ['threshold', 'fpr', 'tpr']    
        arrayOfTPRandFPRByTr = pd.concat([arrayOfTPRandFPRByTr, aux])

    return arrayOfTPRandFPRByTr

#------------------------------------------------------------------------------------------------------------------------------
def getScores(dt, label, folds, clf, proba=True):
    
    skf = StratifiedKFold(n_splits=folds)    
    results = []
    class_labl = []
        
    for train_index, valid_index in skf.split(dt, label):
      tr_data, valid_data = dt[train_index], dt[valid_index]
      tr_lbl = label[train_index]        
      valid_lbl = label[valid_index]        
      clf.fit(tr_data, tr_lbl)  
      if proba:      
          results.extend(clf.predict_proba(valid_data)[:,1])
      else:
          results.extend(clf.decision_function(valid_data))

      class_labl.extend(valid_lbl)
    
    scr = pd.DataFrame(results,columns=["score"])
    scr_labl = pd.DataFrame(class_labl, columns= ["label"])
    scores = pd.concat([scr,scr_labl], axis = 1, ignore_index= False)
    
    return scores

#------------------------------------------------------------------------------------------------------------------------------
def load_data(path="./data/", dts="concrete"):
    prep_file = path + dts + "/prep.py"
    spec = importlib.util.spec_from_file_location("prep", prep_file)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
       
    return prep.prep_data()

#------------------------------------------------------------------------------------------------------------------------------

def get_batch (label, alpha, X_test, y_test, n):
    
    n_pos = int(np.round(n*alpha,0))
    n_neg = n - n_pos     
    i_pos = np.where(y_test == label)
    X_pos = X_test[i_pos,:][0]
    X_pos = X_pos[np.random.randint(X_pos.shape[0], size=n_pos),:]
    y_pos = np.full((1,n_pos), label)[0]
    
    X_neg = X_test[np.where(y_test != label)[0],:]
    dist_neg = np.random.uniform(0,1, len(np.unique(y_test))-1)    
    dist_neg = dist_neg/np.sum(dist_neg)    
    neg_labels = y_test[y_test != label] 

    if alpha == 1.0:
        dist_neg = np.array(dist_neg*(1-alpha))
        test = X_pos
        dist_cl = np.append(dist_neg, alpha)[np.argsort(np.append(np.unique(neg_labels), label))]
        return test, None, dist_cl    
        
    neg_n = np.random.choice(np.unique(neg_labels), n_neg, p=dist_neg)
    labels, values = zip(*Counter(neg_n).items())   
    test = np.empty((0, X_neg.shape[1]))
    for i in range(0,len(labels)):        
        aux = X_neg[np.where(neg_labels == labels[i])[0],:]
        test = np.concatenate([test,aux[np.random.randint(aux.shape[0], size=values[i]),:]])

    test = np.concatenate([test,X_pos])  
    dist_neg = np.array(dist_neg*(1-alpha))     
    dist_cl = np.append(dist_neg, alpha)[np.argsort(np.append(np.unique(neg_labels), label))]   
    return test, np.append(np.array(neg_n), np.array(y_pos)), dist_cl

#------------------------------------------------------------------------------------------------------------------------------    
"""This function fit a quantifier using the codes provided by Tobias Schumacher.
 
Parameters
----------
qntMethod : string
    Quantification method name, according to the alg_index.csv file
X_train : DataFrame
    A DataFrame of the training data.
y_train : DataFrame
    A DataFrame with the training labels.
Returns
-------
object
    the quantifier fitted. 
 """
def fit_quantifier_schumacher_github(qntMethod, X_train, y_train):
    
    algorithm_index = pd.read_csv("./alg_index.csv",
                                sep=";",
                                index_col="algorithm")


    algorithm_index = algorithm_index.loc[algorithm_index.export == 1]
    algorithms = list(algorithm_index.index)

    algorithm_dict = dict({alg: helpers.load_class(algorithm_index.loc[alg, "module_name"],
                                                algorithm_index.loc[alg, "class_name"])
                        for alg in algorithms})

    init_args = []
    fit_args = [np.asarray(X_train), np.asarray(y_train)]   
    qf = algorithm_dict[qntMethod](*init_args)
    qf.fit(*fit_args)

    return qf   

#------------------------------------------------------------------------------------------------------------------------------
"""This function predict the class distribution from a given test set.
 
Parameters
----------
qnt : object
    A quantifier previously fitted from some training data.
X_train : DataFrame
    A DataFrame of the test data.
Returns
-------
array
    the class distribution of the test calculated according to the qntMethod quantifier. 
 """
def predict_quantifier_schumacher_github(qnt, X_test):
    return qnt.predict(*[np.asarray(X_test)])

#------------------------------------------------------------------------------------------------------------------------------
# BEGIN - Load fitted quantifiers or fit them. Finally save them (NOTE: only for T. Schumacher codes)
def fit_quantifiers_SCH(l_trained_qnt, list_sch, X_train, y_train):
    
    quantifier_added = False
    if l_trained_qnt is not None:
        for qnt in list_sch:
            counter = str.split(qnt, 'SCH_')[1]            
            if counter not in [ co['counter'] for co in l_trained_qnt]:                
                qnt_sch = fit_quantifier_schumacher_github(counter, X_train, y_train)
                l_trained_qnt.append({"counter":counter, "model": qnt_sch})
                print('Treinou!')
                quantifier_added = True
        
        return l_trained_qnt, quantifier_added
        
    else:
        for qnt in list_sch:
            counter = str.split(qnt, 'SCH_')[1]            
            qnt_sch = fit_quantifier_schumacher_github(counter, X_train, y_train)
            l_trained_qnt.append({"counter":counter, "model": qnt_sch})
        return l_trained_qnt
#------------------------------------------------------------------------------------------------------------------------------
def load_and_fit_schumacher_quantifiers(X_train, y_train, counters, path_dts, model_setup, clf_name='SVC'):    
    list_sch = [match for match in counters if "SCH_" in match]  
    if len(list_sch) > 0:
        l_qnt_sch = []
    else:
        return None
    if os.path.isfile(path_dts+'schumacher/'+model_setup+'_'+ clf_name +'_sch_models.sav') is True:
        l_qnt_sch = pickle.load(open(path_dts+'schumacher/'+model_setup+'_'+ clf_name +'_sch_models.sav', 'rb'))
        print('Schumacher quantifiers loaded ')
    
    #list_existing = [x['counter'] for x in l_qnt_sch]
    l_qnt_sch, quantifier_added = fit_quantifiers_SCH(l_qnt_sch, list_sch, X_train, y_train)
    if quantifier_added:
        pickle.dump(l_qnt_sch, open(path_dts+'schumacher/'+model_setup+'_'+ clf_name +'_sch_models.sav', 'wb'))
    return l_qnt_sch
  # END - Load fitted quantifiers or fit them        

#------------------------------------------------------------------------------------------------------------------------------
def load_and_fit_quantifiers_binary(X_train, y_train, allow_proba, model_setup, n_classes, clf, path_dts, clf_name):

  if os.path.isfile(path_dts+'binary/'+model_setup+'_'+clf_name+'_models_bin.sav') is False:
    models = []
    l_scores = []
    
    lbin = preprocessing.LabelBinarizer()
    y_bin = lbin.fit_transform(y_train)
    for ci in range(0,n_classes):      
      clf.fit(X_train, y_bin[:,ci])
      models.append(deepcopy(clf))
      l_scores.append(getScores(X_train, y_bin[:,ci], 10, clf, allow_proba))
      #l_train.append(X_train)
      #l_ytrain.append(y_bin[:,ci])      

    pickle.dump(models, open(path_dts+'binary/'+model_setup+'_'+clf_name+'_models_bin.sav', 'wb')) # Loading fitted models
    pickle.dump(l_scores, open(path_dts+'binary/'+model_setup+'_'+clf_name+'_list_scores_bin.sav', 'wb')) # Loading scores
    #pickle.dump(l_train, open(path_dts+'l_train.sav', 'wb')) # Loading training partition
    #pickle.dump(l_ytrain, open(path_dts+'l_ytrain.sav', 'wb')) # Loading labels from training partition
  else:
    models = pickle.load(open(path_dts+'binary/'+model_setup+'_'+clf_name+'_models_bin.sav', 'rb'))
    l_scores = pickle.load(open(path_dts+'binary/'+model_setup+'_'+clf_name+'_list_scores_bin.sav', 'rb'))
    print('OVR classifiers loaded!')
    #l_train = pickle.load(open(path_dts+'l_train.sav', 'rb'))
    #l_ytrain = pickle.load(open(path_dts+'l_ytrain.sav', 'rb')) 

  return models, l_scores

#------------------------------------------------------------------------------------------------------------------------------
def load_and_fit_quantifiers_multiclass(X_train, y_train, model_setup, clf, path_dts, clf_name):
  if os.path.isfile(path_dts+'multiclass/'+model_setup+'_'+clf_name+'_models_MC.sav') is False:
    print('Estimating scorer and scores for multiclass setup')
    clf.fit(X_train, y_train)
    Y_cts = np.unique(y_train, return_counts=True)
    nfolds = min(10, min(Y_cts[1]))
    
    train_scores = np.zeros((len(X_train), len(np.unique(y_train))))
    train_labels = np.zeros(len(X_train))

    classes = np.sort(np.unique(y_train))

    model = deepcopy(clf)   
    if nfolds > 1:
        kfold = StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)
        for train, test in kfold.split(X_train, y_train):
            model.fit(X_train[train], y_train[train])
            train_scores[test] = model.predict_proba(X_train)[test]
            train_labels[test] = np.int0(y_train[test])

        scr = pd.DataFrame(train_scores, columns= classes)
        scr_labl = pd.DataFrame(train_labels, columns= ["label"])
        scores = pd.concat([scr,scr_labl], axis = 1, ignore_index= False)
    else:
        train_scores[test] = clf.predict_proba(X_train)
        train_labels[test] = np.int0(y_train)
        scr = pd.DataFrame(train_scores, columns= classes)
        scr_labl = pd.DataFrame(train_labels, columns= ["label"])
        scores = pd.concat([scr, scr_labl], axis = 1, ignore_index= False) 

    pickle.dump(scores, open(path_dts+'multiclass/'+model_setup+'_'+clf_name+'_scores_MC.sav', 'wb')) # Loading fitted models
    pickle.dump(clf, open(path_dts+'multiclass/'+model_setup+'_'+clf_name+'_models_MC.sav', 'wb')) # Loading fitted models    
  else:
    print(clf_name + ' loaded!')
    clf = pickle.load(open(path_dts+'multiclass/'+model_setup+'_'+clf_name+'_models_MC.sav', 'rb'))
    scores = pickle.load(open(path_dts+'multiclass/'+model_setup+'_'+clf_name+'_scores_MC.sav', 'rb'))
  return clf, scores