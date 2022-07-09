# -*- coding: utf-8 -*-

import pandas as pd

import io
import requests

def prep_data(binned = False):

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

  s = requests.get(url)

  colnames = ["comp" + str(i+1) for i in range(8)] + ["strength"]

  output = open('./data/concrete/dta.xls', 'wb')
  output.write(s.content)
  output.close()
  
  dta = pd.read_excel('./data/concrete/dta.xls',
              header = 0,
              names = colnames)
  
  bins = [0, 20, 35, 100]
  labels = [1,2,3]
  dta['label'] = pd.cut(dta['strength'], bins = bins, labels = labels)
  dta['label'] = dta['label'].astype("int64")

  if binned:
    for col in list(dta)[:-1]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
