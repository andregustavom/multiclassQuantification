# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "./datasets/Video_Games_Sales_as_at_22_Dec_2016.csv"

  dta = pd.read_csv(url,
              header = 0,
              skipinitialspace = True)
  
  dta = dta.dropna()
  dta = dta.drop(["Name","Publisher","Global_Sales","Developer"], axis = 1)
  dta = pd.get_dummies(dta)
  
  bins = [0, 50, 70, 80, 100]
  labels = [0,1,2,3]
  dta['label'] = pd.cut(dta['Critic_Score'], bins = bins, labels = labels)
  dta['label'] = dta['label'].astype("int64")
  
  if binned:
    for col in ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Count', 'User_Count']:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
