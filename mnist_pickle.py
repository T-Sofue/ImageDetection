from math import *
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('mnist_test.csv')
x=data.iloc[:,1:785]
y=data.iloc[:,0]
clf=RandomForestClassifier(random_state=0,n_estimators=250, min_samples_split=2)
clf.fit(x,y)
test=pd.read_csv('mnist_train.csv')
x=test.iloc[:,1:785]
y=test.iloc[:,0]
print(clf.score(x,y))

with open('pickle.sav', 'wb') as f:
    cPickle.dump(clf, f)
