import pandas as pd
import numpy as np

import pdb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pickle
import os.path

import sys

import warnings
warnings.filterwarnings("ignore")


import utils.auxiliary as utils


from utils.quantifierTemplate import apply_quantifier



from sklearn import svm

from copy import deepcopy



def run_expereiment(X_train, X_test, y_train, y_test, dts_name, models, l_scores):
  #>>>>>>>..............Experimental_setup............>>>>>>>>>>
  vdist = ["topsoe", "jensen_difference", "prob_symm", "ord", "sord", "hellinger"] 
  names_vdist = ["TS", "JD", "PS", "ORD", "SORD", "HD"] 
  counters    = ['DySyn','DySyn+aMoSS','DyS-TS', 'SCH_GPAC', 'SCH_DyS']
  measure     = "topsoe"                   #default measure for DyS
  niterations = 10
  alpha_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

  n_classes = np.sort(np.int0(np.unique(y_train)))
  print(n_classes)
  y_test = y_test.astype(int)

  #clf = RandomForestClassifier(n_estimators=200)
  clf= svm.SVC()
  
  if models is None:

    list_sch = [match for match in counters if "SCH_" in match]  
    if len(list_sch) > 0:
        l_qnt_sch = []
        for schi in list_sch:
            qnt_sch = utils.fit_quantifier_schumacher_github(schi.split('_')[1], X_train, y_train)
            l_qnt_sch.append({"counter":schi.split('_')[1], "model": qnt_sch})
        
        pickle.dump(l_qnt_sch, open('./models_scores/'+dts_name+'/sch_models.sav', 'wb'))

    models = []
    l_scores = []
    l_train = []
    l_ytrain= []
    lbin = preprocessing.LabelBinarizer()
    y_bin = lbin.fit_transform(y_train)
    for ci in range(0,len(n_classes)):      
      #train_bin, y_bin = training_OVR(np.array(X_train), y_train, ci)
      clf.fit(X_train, y_bin[:,ci])
      models.append(deepcopy(clf))
      l_scores.append(utils.getScores(X_train, y_bin[:,ci], 10, clf))
      l_train.append(X_train)
      l_ytrain.append(y_bin[:,ci])      

    pickle.dump(models, open('./models_scores/'+dts_name+'/models.sav', 'wb'))
    pickle.dump(l_scores, open('./models_scores/'+dts_name+'/l_scores.sav', 'wb'))
    pickle.dump(l_train, open('./models_scores/'+dts_name+'/l_train.sav', 'wb'))
    pickle.dump(l_ytrain, open('./models_scores/'+dts_name+'/l_ytrain.sav', 'wb'))
  else:
    models = pickle.load(open('./models_scores/'+dts_name+'/models.sav', 'rb'))
    l_scores = pickle.load(open('./models_scores/'+dts_name+'/l_scores.sav', 'rb'))
    l_train = pickle.load(open('./models_scores/'+dts_name+'/l_train.sav', 'rb'))
    l_ytrain = pickle.load(open('./models_scores/'+dts_name+'/l_ytrain.sav', 'rb'))
    l_qnt_sch = pickle.load(open('./models_scores/'+dts_name+'/sch_models.sav', 'rb'))
     
  result = pd.DataFrame()   
  
  batch_sizes = [50]  
    
  for sample_size in batch_sizes:   #[10,100,500], batch_sizes, Varying test set sizes 
    for cl in n_classes:
      for alpha in alpha_values: 
        print(str(cl) + ' - ' +str(alpha))       
        for iter in range(niterations):
          #print('Sample size #%d' % (sample_size))
          #print('iteration #%d' % (iter + 1))
          sample_test, _, prop_actual = utils.get_batch(cl, alpha, X_test, y_test, sample_size) 
          #print(prop_actual)

          for co in counters:
            auxi = co.split('_')
            quantifier = co
            if len(auxi) > 1:
                auxi = auxi[1]
                l_i_sch = [x['counter']==auxi for x in l_qnt_sch]
                l_qnt_i = np.where(np.array(l_i_sch)== True)[0][0]
                pred_dist = utils.predict_quantifier_schumacher_github(l_qnt_sch[l_qnt_i]['model'], sample_test)  
            else:
                aux = co.split("-")
                pred_dist = []

                if len(aux) > 1:
                    quantifier = aux[0]
                    measure = vdist[names_vdist.index(aux[1])]
            
                for i in range(0, len(models)):
                    clf = models[i]   
                    #te_scores = clf.predict_proba(sample_test)[:,1]  #estimating test sample scores                
                    te_scores = clf.decision_function(sample_test)  #estimating test sample scores               
                    scores = deepcopy(l_scores[i])
                    if quantifier == "DySyn+aMoSS":
                        if (min(scores.score) > 0):  
                            scores.score = scores.score - min(scores.score)
                        else:
                            scores.score = scores.score + abs(min(scores.score))

                    tprfpr = None#qntu.getTPRandFPRbyThreshold(scores)           
                    #.............Calling of Methods..................................................           
                    pos_scores = scores[scores['label']==1]['score']
                    neg_scores = scores[scores['label']==0]['score']

                    #return pos_scores, neg_scores
                    u_p = np.mean(pos_scores)
                    u_n = np.mean(neg_scores)

                    pred_pos_prop = apply_quantifier(qntMethod = quantifier, p_score = pos_scores, n_score = neg_scores, test_score = te_scores, 
                                                    TprFpr = tprfpr, thr = 0.5, measure = measure, calib_clf = None, X_test = sample_test, 
                                                    u_p = u_p, u_n = u_n)         
                
                    pred_pos_prop = round(pred_pos_prop,2)  #predicted class proportion
                    pred_dist.append(pred_pos_prop)                   
                
              #..............................RESULTS Evaluation.....................................
                aux_sum = np.sum(pred_dist)
                if aux_sum != 0.0:
                    pred_dist = pred_dist/aux_sum


            pred_dist = np.round(pred_dist, 3)
            #abs_error = np.round(np.sum(abs(pred_dist - prop_actual))/len(n_classes),3) #absolute error  
            abs_error = np.round(np.sum(abs(pred_dist - prop_actual)),3)
            #abs_error = qp.error.absolute_error(pred_dist, prop_actual)    
            #print(abs_error)

            line_result = pd.concat([pd.DataFrame([sample_size, abs_error, quantifier, dts_name]).T, pd.DataFrame(np.array(prop_actual)).T, pd.DataFrame(np.array(pred_dist)).T], axis=1)

            result = pd.concat([result, line_result], axis=0)

    co_names = ["Test_size","abs_error","quantifier", "dataset"]
    co_actual = ["actual_c."+ str(x) for x in np.unique(n_classes)]
    co_pred = ["pred_c."+ str(x) for x in np.unique(n_classes)]
    co_names.extend(co_actual)
    co_names.extend(co_pred) 

    result.columns = co_names  

  return result


def run(dt):

  print(dt)
  models = None
  if os.path.isfile('./models_scores/'+dt+'/result.csv') is False:
    if os.path.isfile('./models_scores/'+dt+'/models.sav'):
      models = 1
    
    df_data = utils.load_data(path="./data/", dts=dt)
    X = np.array(df_data.drop(['label'], axis=1))
    #y = np.int0(df_data['Class'])
    y = np.array(df_data['label'])
    #y = y.astype('category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)
    mo = run_expereiment(X_train, X_test, y_train, y_test, dt, models, 1)
    #mo_ant = pd.read_csv('./models_scores/'+dt+'/result.csv')
    #mo_ant = mo_ant.drop(mo_ant[mo_ant.quantifier=='DySyn+aMoSS'].index)

    #mo = pd.concat([mo, mo_ant], axis=0)
    
    mo.to_csv('./models_scores/'+dt+"/result.csv")



if __name__ == "__main__":
  run(sys.argv[1])