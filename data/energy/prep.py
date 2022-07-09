# -*- coding: utf-8 -*-

import pandas as pd

import io
import requests

def prep_data(binned = False):

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

  s = requests.get(url).content

  dta = pd.read_csv(io.StringIO(s.decode('utf-8')),
                na_values=['?'],
                skipinitialspace = True)

  
  dta = dta.drop(["date", "rv1","rv2"],axis = 1)
  
  bins = [0, 50, 100,2000]
  labels = [1,2,3]
  dta['label'] = pd.cut(dta['Appliances'], bins = bins, labels = labels)
  dta['label'] = dta['label'].astype("int64")

  if binned:
    for col in list(dta)[1:]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")