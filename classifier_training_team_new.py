###########################################################################
''' Import the required libraries '''

#Using the below given methods is one way to train your classifier
#Define your classifier in this file
###########################################################################
import pandas as pd 
def divideData(originalData):
   import random
   TRAIN_TEST_RATIO = 0.8
   train = []
   test = []
   for one in originalData:
       if random.random() < TRAIN_TEST_RATIO:
           train.append(one)  
       else:
           test.append(one)
   return train, test

import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from Interactive_voice_sample_collector14 import wavveclive
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
wavvec1t = wavveclive

trainData, testData = divideData(wavvec1t)
trainData = np.array(trainData)
testData = np.array(testData)
train_featureVec,train_labels =zip(*trainData)
test_featureVec,test_labels = zip(*testData)
train_featureVec1 = list(train_featureVec)
test_featureVec1 = list(test_featureVec)
train = pd.DataFrame(train_featureVec1)
test = pd.DataFrame(test_featureVec1)

train_labels1 = list(train_labels)
test_labels1 = list(test_labels)
train_y = pd.DataFrame(train_labels1).astype('int')
test_y = pd.DataFrame(test_labels1).astype('int')

#for j in range(1,30,2):
#    clf = DecisionTreeClassifier(max_depth = j)
#    model = clf.fit(train, train_y)
#    pred_test = model.predict(test)
#    print(accuracy_score(pred_test, test_y),j)
#
#for k in range(1,30):
#    k=k+2
#    neigh =  KNeighborsClassifier(n_neighbors=k)
#    model1 = neigh.fit(train, train_y)
#    pred_test_KNN = model1.predict(test)
#    KN= accuracy_score(pred_test_KNN, test_y)
#    print(KN,k)

clf = DecisionTreeClassifier(max_depth = 15)
model_ID3 = clf.fit(train, train_y)
pred_test = model_ID3.predict(test)
print("accuracy with ID3:",accuracy_score(pred_test, test_y))


neigh =  KNeighborsClassifier(n_neighbors=5)
model_KNN = neigh.fit(train, train_y)
test1=[]
test1=test1.append(test[0])
pred_test_KNN = model_KNN.predict(test)
#print(pred_test_KNN)
#cd = model_KNN.predict_proba(test)
#print(max(cd[0]))
KN= accuracy_score(pred_test_KNN, test_y)
print("accuracy with KNN:",KN)



