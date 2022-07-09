# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:45:02 2018

@author: tobi_
"""
import io
import requests

import pandas as pd

def prep_data():


  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00262/turkiye-student-evaluation_generic.csv"

  s = requests.get(url).content

  dta = pd.read_csv(io.StringIO(s.decode('utf-8')),
                sep = ',',
                index_col = False,
                skipinitialspace = True)

  
  dta = dta.drop(columns = ['class','nb.repeat'])
  
  dta = dta.rename(columns = {'instr':'label'})

  #dta.to_pickle("dta.pkl")
  
  return dta