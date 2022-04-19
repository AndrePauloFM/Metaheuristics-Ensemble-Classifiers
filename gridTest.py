
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

"""Os resultados de cada classificador devem ser apresentados numa tabela contendo a média das acurácias obtidas em cada fold do ciclo externo, o desvio padrão e o intervalo de confiança a 95% de significância dos resultados, e também através do boxplot dos resultados de cada classificador em cada fold.

# Functions Grid and Results

Os dados utilizados no conjunto de treino em cada rodada de teste são padronizados (normalizados o com z-score). Os valores de padronização obtidos nos dados de treino são utilizados para padronizar os dados do respectivo conjunto de teste.
O procedimento experimental de treinamento, validação e teste é realizado através de 3 rodadas de ciclos aninhados de validação e teste, com o ciclo interno de validação contendo 4 folds e o externo de teste com 10 folds. A busca em grade (grid search) do ciclo interno considera os os valores de hiperparâmetros definidos para cada técnica de aprendizado.
"""

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
  # Acurácia Metaheurística
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


