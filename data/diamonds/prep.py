# -*- coding: utf-8 -*-

import pandas as pd

import io
import requests

def prep_data(binned = False):

  url = "./datasets/diamonds.csv"

  dta = pd.read_csv(url,
              header = 0,
              index_col = 0,
              skipinitialspace = True)
  
  dta.cut = dta.cut.replace({"Fair": 0, 
                             "Good": 0, 
                             "Very Good": 0, 
                             "Ideal": 2, 
                             "Premium": 1})

  dta = dta.rename(columns={"x": "xc", "y": "yc", "z": "zc", "cut":"label"})

  dta = pd.get_dummies(dta)
  
  if binned:
    for col in ['carat','depth', 'table', 'price', 'xc', 'yc', 'zc']:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
