# -*- coding: utf-8 -*-

import pandas as pd
"""# Class - Heterogeneous Polling"""
import heterogeneousClassifier as HP
"""# Get Results Function"""
import gridTest as gt
"""# import Metaheuristicas"""
import metaheuristicas as MT

dfResultClassifier = pd.DataFrame(columns=['Métodos', 'Média', 'STD', 'Limite Inferior', 'Limite Superior'])
dfResultClassifier['Métodos'] = ['HillClimbing', 'Simulated Annealing', 'Genectic Algorithm']


def evaluateMetaheuristicas(dataBase):
  """### Hill Climbing"""

  # Performe Hill Climbing
  print('Performing Hill Climbing ...')
  optimal_state, optimal_value = MT.hill_climbing(dataBase, 600)

  # Test optimal state
  sampleSize = int(len(optimal_state)/3)
  grid = {'estimator__n_samples': [sampleSize]}
  model = HP.HeterogeneousClassifier(currentState=optimal_state)
  print('\n Model Evaluate... Heterogeneous Classifier: ', optimal_state)
  # Model Evaluate
  hcScores, acc = gt.GridTestModel(dataBase, model, grid)

  # Results
  print('****** Results ******\n')
  print('DataBase: ', dataBase.DESCR[4:18])
  print('Metaheuristic - Hill Climbing')
  print('Best Configuration: ', optimal_state, '\n')
  hc_mean, hc_std, hc_inf, hc_sup = gt.getResults(hcScores)
  print(hcScores)
  print("Mean Accuracy: %0.3f Standard Deviation: %0.3f" % (hc_mean, hc_std))
  print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
          (hc_inf, hc_sup))

  dfResultClassifier.iloc[0] = ['HillClimbing', hc_mean, hc_std, hc_inf, hc_sup]
  dfResultClassifier

  """### Simulated Annealing"""

  # Simulated Annealing
  max_time = 1800
  inter_max = 20
  t = 200
  alfa = 0.1
  print('Performing Simulated Annealing...')
  optimal_state, optimal_value = MT.simulated_annealing(dataBase,max_time,inter_max,t,alfa)

  # Test Optimal State Simulated Annealing
  sampleSize = int(len(optimal_state)/3)
  grid = {'estimator__n_samples': [sampleSize]}
  model = HP.HeterogeneousClassifier(currentState=optimal_state)
  print('\n Model Evaluate... Set of Classifiers Selected: ', optimal_state)
  # Model Evaluate
  saScores, acc = gt.GridTestModel(dataBase, model, grid)

  # Resultados
  print('****** Results ******\n')
  print('DataBase: ', dataBase.DESCR[4:18])
  print('Metaheuristic - Simulated Annealing')
  print('Best Configuration: ', optimal_state, '\n')
  sa_mean, sa_std, sa_inf, sa_sup = gt.getResults(saScores)
  print(saScores)
  print("Mean Accuracy: %0.3f Standard Deviation: %0.3f" % (sa_mean, sa_std))
  print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
          (sa_inf, sa_sup))

  dfResultClassifier.iloc[1] = ['Simulated Annealing', sa_mean, sa_std, sa_inf, sa_sup]
  dfResultClassifier

  """### Genetic Algorithm"""

  # define the total iterations
  n_iter = 8
  # bits
  n_samples = [3,5,7]
  # define the population size
  n_pop = 10
  # crossover rate
  r_cross = 0.9
  # mutation rate
  r_mut = 1.0 / float(n_samples[0])
  # perform the genetic algorithm search
  print('Performing Genetic Algorithm ...')
  optimal_state, optimal_value = MT.genetic_algorithm(dataBase, MT.objetiveFunction, n_samples, n_iter, n_pop, r_cross, r_mut)
  print('Done!')
  print('f(%s) = %f' % (optimal_state, optimal_value))

  # test Optimal State
  sampleSize = int(len(optimal_state)/3)
  grid = {'estimator__n_samples': [sampleSize]}
  model = HP.HeterogeneousClassifier(currentState=optimal_state)
  print('\n Model Evaluate... Conjunto de classificadores selecionados: ', optimal_state)
  # Model Evaluate
  gaScores, acc = gt.GridTestModel(dataBase, model, grid)

  # Resultados
  print('****** Results ******\n')
  print('DataBase: ', dataBase.DESCR[4:18])
  print('Metaheuristic - Genetic Algorithm')
  print('Best Configuration: ', optimal_state, '\n')
  ga_mean, ga_std, ga_inf, ga_sup = gt.getResults(gaScores)
  print(gaScores)
  print("Mean Accuracy: %0.3f Standard Deviation: %0.3f" % (ga_mean, ga_std))
  print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
          (ga_inf, ga_sup))

  """## Resultados Metaheurísticas Acurácia"""

  dfResultClassifier.iloc[2] = ['Genectic Algorithm', ga_mean, ga_std, ga_inf, ga_sup]
  dfResultClassifier

  import plotly.graph_objects as go
  scores = [hcScores, saScores, gaScores]
  scoresNames = ['Hill Climbing', 'Simulated Annealing', 'Genetic Algorithm']
  fig = go.Figure()
  for i in range(len(scores)):
    fig.add_trace(go.Box(y=scores[i], name=scoresNames[i]))
  fig.update_layout(
      yaxis_title='Acurácia',
      xaxis_title='Metaheurísticas',
      title=dataBase.DESCR[4:18]+'Desempenho dos Metaheurísticas  - Acurácia',
  )
  fig.show()

  """## Paired t-test and Wilcoxon Test"""

  from scipy.stats import ttest_rel, wilcoxon
  scores = [hcScores, saScores, gaScores]
  scoresNames = ['Hill Climbing','Simulated Annealing', 'Genetic Algorithm']
  dfPairTest = pd.DataFrame(columns=[0,1,2])
  for i in range(len(scores)):
    for j in range(len(scores)):
      if j == i:
        dfPairTest.at[i, j] = scoresNames[i]
      
      if j > i:
        print('Paired t-test', scoresNames[i], scoresNames[j])
        s,p = ttest_rel(scores[i],scores[j])
        print("t: %0.2f p-value: %0.8f\n" % (s,p))
        dfPairTest.at[i, j] = p

      if j < i :
        print ('Wilcoxon Test', scoresNames[i], scoresNames[j])
        s,p = wilcoxon (scores[i],scores[j])
        print("w: %0.2f p-value: %0.8f\n" % (s,p))
        dfPairTest.at[i, j] = p

  dfPairTest.columns = ['T1','T2','T3']
  dfPairTest.index = ['w1', 'w2', 'w3']
  dfPairTest
