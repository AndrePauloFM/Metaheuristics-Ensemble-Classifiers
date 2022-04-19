import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import randint
from numpy.random import rand
import time
import itertools

def GridTestModel(dataBase, model, grid):

  # Data Base
  X = dataBase.data
  y = dataBase.target

  # Z-score
  scalar = StandardScaler()

  # Pipeline
  pipeline = Pipeline([('transformer', scalar), ('estimator', model)])

  # configure the cross-validation procedure
  rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234)

  # configure Grid Search
  gs = GridSearchCV(estimator=pipeline, param_grid = grid, 
                    scoring='accuracy', cv = 4, verbose=0, refit=True)

  # Results
  scores = cross_val_score(gs, X, y, scoring='accuracy', 
                          cv = rkf)
  # Accuracy Metaheuristc
  acuracia = model.metaheuristica_acuracia
  return scores, acuracia

def getResults(scores):
  # print (scores)
  mean = scores.mean()
  std = scores.std()
  inf, sup = stats.norm.interval(0.95, loc=mean, 
                                scale=std/np.sqrt(len(scores)))
  # print("Mean Accuracy: %0.3f Standard Deviation: %0.3f" % (mean, std))
  # print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
  #       (inf, sup)) 
  return mean, std, inf, sup
