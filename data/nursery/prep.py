# -*- coding: utf-8 -*-

import pandas as pd

import io
import requests

def prep_data():

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"

  colnames = ["att" + str(i+1) for i in range(8)]
  colnames.append("label")


  s = requests.get(url).content


  dta = pd.read_csv(io.StringIO(s.decode('utf-8')),
                names = colnames,
                header=None,
                skipinitialspace = True)
  
  dta.label = dta.label.replace({"not_recom"  : 0, "recommend" : 1, "very_recom" : 1, "priority": 1, "spec_prior" : 2})
  dta = pd.get_dummies(dta)
  
  return dta

#dta.to_pickle("dta.pkl")

