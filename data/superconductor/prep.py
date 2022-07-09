# -*- coding: utf-8 -*-


from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd

import certifi


def prep_data(binned = False):

  url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip", cafile=certifi.where())
  
  my_zip_file = ZipFile(BytesIO(url.read()))
  f = my_zip_file.namelist()[1]
  
  dta =  pd.read_csv(my_zip_file.open(f),
                  header = 0,
                  skipinitialspace = True)

  bins = [0, 5,20, 60, 2000]
  labels = [0, 1, 2, 3]
  dta['label'] = pd.cut(dta['critical_temp'], bins=bins, labels=labels)
  dta['label'] = dta['label'].astype("int64")
  dta = pd.get_dummies(dta, columns = ["number_of_elements"])

  if binned:
    for col in list(dta)[:-10]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")



  #dta.to_pickle("dta.pkl")
  #dta.to_pickle("dta_binned.pkl")
  
  return dta