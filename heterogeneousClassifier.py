
import numpy as np


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.utils import resample
from collections import Counter


from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, clone
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import randint
from numpy.random import rand
import time

import itertools


"""# Classe - Heterogeneous Polling"""

from sklearn.metrics import accuracy_score
def mostCommon(estimatorsPredict):
  
    return [Counter(col).most_common() for col in zip(*estimatorsPredict)]
  

def votingClass(predMatrix, y_train):

    saida = np.array([])
    
    valuesFrequency = mostCommon(predMatrix)
    for j in range(predMatrix.shape[1]):
      if len(valuesFrequency[j])>1 and valuesFrequency[j][0][1] == valuesFrequency[j][1][1]:
        listaElementsTie = np.array([])
        for k in range(len(valuesFrequency[j])-1):
          if valuesFrequency[j][k][1] == valuesFrequency[j][k+1][1]:
            if k == 0 :
              listaElementsTie = np.append(listaElementsTie, valuesFrequency[j][k][0])
              listaElementsTie = np.append(listaElementsTie, valuesFrequency[j][k+1][0])
            else:
              listaElementsTie = np.append(listaElementsTie, valuesFrequency[j][k+1][0])
        # Retorna a classe mais votada mais frequente na base de treino
        saida = np.append(saida, compareFrequencyValues(y_train, listaElementsTie))
      else:
        # Retorna a classe mais votada
        saida = np.append(saida, valuesFrequency[j][0][0])
    return saida

def compareFrequencyValues(y_train, elementsTie):
      indexElements = []
      orderArray = getMostfrequentValues(y_train)
      listValues = list(orderArray)
      for i in range(len(elementsTie)):
        indexElements.append(listValues.index(elementsTie[i]))
      mostFreq = min(indexElements)
      return orderArray[mostFreq]

def getMostfrequentValues(a):

    from collections import Counter
    mostfrequentValues = np.array([])
    b = Counter(a)
    arrayValoresFreq = b.most_common()
    for i in range(len(b)):
      mostfrequentValues = np.append(mostfrequentValues,arrayValoresFreq[i][0])
    return mostfrequentValues   


class HeterogeneousClassifier(BaseEstimator):
  
  estimatorBase = list()
  estimatorBase.append(KNeighborsClassifier(n_neighbors=1))
  estimatorBase.append(DecisionTreeClassifier())
  estimatorBase.append(GaussianNB())
  metaheuristica_acuracia = []

  def __init__(self, base_estimator = estimatorBase,n_samples=None, currentState=None):
    
    self.base_estimator = base_estimator
    self.n_samples = n_samples
    self.ord = []
    self.estimators = []
    self.KNNclassifier = KNeighborsClassifier(n_neighbors=1)
    self.DTclassifier = DecisionTreeClassifier()
    self.NBclassifier = GaussianNB()
    self.currentState = currentState
    self.estimatorsStateVector = []

    self.x_train_original = []
    self.y_train_original = []
    self.acuracia = []


  def fit(self, X, y):
    self.x_train_original = X
    self.y_train_original = y

    self.estimatorsStateVector = self.currentState
    self.ord = y
    for i in range(self.n_samples):

        X, y = resample(X,y, replace=True, random_state=i)

        if self.estimatorsStateVector[0+(3*i)] == 1:
          self.estimators.append(self.KNNclassifier.fit(X, y))
        if self.estimatorsStateVector[1+(3*i)] == 1:
          self.estimators.append(self.DTclassifier.fit(X, y))
        if self.estimatorsStateVector[2+(3*i)] == 1:
          self.estimators.append(self.NBclassifier.fit(X, y))
        
    ypred_ClassificadoresCombinados = self.predict(self.x_train_original)
    aux_acuracia = accuracy_score(self.y_train_original, ypred_ClassificadoresCombinados)
    self.acuracia.append(aux_acuracia)
    HeterogeneousClassifier.metaheuristica_acuracia = np.mean(self.acuracia)
   
    return self.estimators



  def predict(self, X):
    y_pred = []
    for i in range(len(self.estimators)):
      pred = np.array([self.estimators[i].predict(X)])

      if i == 0:
        y_pred = np.vstack((pred))
      else:
        y_pred = np.vstack((y_pred, pred))

    y_predVot = votingClass(y_pred,self.ord)
      
    return y_predVot
