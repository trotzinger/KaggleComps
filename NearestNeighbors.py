'''
author: Tim Rotzinger
date: 3/20/2018

This code is an attempt at the kaggle beggener compitiion found on https://www.kaggle.com/c/titanic#description.
This file will be to try a nearest neighbors approach.
'''

import pandas as pd
import scipy as sp
import numpy as np

def getData(path):
    return pd.read_csv(path, sep=',', decimal='.', header=0, names=['PassengerId','Survived','Pclass','Name','Sex','Age',
                                                                       'SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
if __name__=="__main__":
    data = getData("C:/projects/KaggleComps/trainData/train.csv")
    data = data.drop('PassengerId', 1)
    
    print ("PANDAS!")