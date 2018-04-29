'''
author: Tim Rotzinger
date: 3/20/2018

This code is an attempt at the kaggle beggener compitiion found on https://www.kaggle.com/c/titanic#description.
This file will be to try a naive bayes approach.
'''

import pandas as pd
import scipy as sp
import numpy as np
from sklearn import naive_bayes
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from random import randint


def getData(path):
    return pd.read_csv(path, sep=',', decimal='.', header=0, names=['PassengerId','Survived','Pclass','Name','Sex','Age',
                                                                    'SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
if __name__=="__main__":
    data = getData("C:/projects/KaggleComps/trainData/train.csv")
    print(data.head())
    #data = data.drop('PassengerId', 1)
    binarySex = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    print(binarySex.values)
    print(data.Survived)
    #cleanAge = data['Age'].apply(lambda x: 0 if (x >= 200 or x < 0 or x == 'NaN' or x is None) else x)
    cleanAge = data['Age'].fillna(value=data.Age.mean())
    sexClean = binarySex.fillna(value=(randint(0,1)))
    print (cleanAge)
    d = {'sex':sexClean, 'age':cleanAge, 'fare':data.Fare, 'Survived':data.Survived}
    xDf = pd.DataFrame(d)
    print (xDf)
    #xDf = pd.DataFrame(xDf['Sex'].fillna(value=randint(0,1)), cleanAge)
    #print (xDf)
    
    #X = np.array(binarySex.values)
    X = np.array(xDf)
    #X = X.reshape(1, 2) #reshape for single feature
    y = np.array(data['Survived'])
    #y = y.reshape(-1, 1) #reshape for single feature
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,
                                                        random_state=42)
    trainNN = naive_bayes.GaussianNB()
    trainNN.fit(X_train,y_train)
    pred = trainNN.predict(X_test)
    print (accuracy_score(y_test, pred))
    print ("PANDAS!")