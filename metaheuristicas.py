
import numpy as np

from numpy.random import randint
from numpy.random import rand
import time

import itertools

"""# Classe - Heterogeneous Polling"""
import heterogeneousClassifier as HP

"""# Get Results Function"""
import gridTest as gt



"""# Metaheurísticas"""

def generatePossibleStates(currentState):
  
  possibleStates = []
  interacoes = np.sum(currentState)
  n=len(currentState)
  lst = list(itertools.product([0, 1], repeat=n))
  for i in range(len(lst)):
    somaElementos=np.sum(np.bitwise_or(currentState,lst[i]))
    if somaElementos == interacoes+1:
      possibleStates.append(np.bitwise_or(currentState,lst[i]))
  possibleStates = set(map(tuple, possibleStates))
  possibleStates = np.array(list(possibleStates))
  return possibleStates

def objetiveFunction (dataBase, currentState, sample):
    # check no zero vector state
    if np.sum(currentState) == 0:
        currentState[0] = 1
    # Define the model with current state
    model = HP.HeterogeneousClassifier(currentState=currentState)
    grid = {'estimator__n_samples': [sample]}
    # Model Evaluate
    hpScores, acuracia = gt.GridTestModel(dataBase, model, grid)
    return acuracia

"""## Hill Climbing Function"""

def hill_climbing(dataBase, max_time):
    # Define n_samples
    nsamples = [3,5,7]

    # Define Process time
    start = time.process_time()
    end = time.process_time()
    ElapseTime = end-start

    # define the model
    model = HP.HeterogeneousClassifier()
    
    # Inicial Values
    optimal_value = 0
    optimal_state = []

    # Evaluate All n_samples
    for sample in range(len(nsamples)):
        flagOtptimal = True
        valid_states = [1]
        estimatorsVectorSize = nsamples[sample]*3

        # Create inicial State
        inicialState = [1]
        while len(inicialState) < estimatorsVectorSize:
          inicialState = np.append(inicialState,0)
        current_state = inicialState

        # Evaluate the model for inicial state
        print('\n Model Evaluate... Hill Climbling: ', current_state)
        metaheuristica_acuracia = objetiveFunction(dataBase,current_state, nsamples[sample])
        aux_val = metaheuristica_acuracia
        print("Mean Accuracy: %0.3f" % (metaheuristica_acuracia))
        if aux_val > optimal_value:
            optimal_value = aux_val
            optimal_state = current_state
            flagOtptimal = True
       
        while len(valid_states) > 0 and flagOtptimal == True:
          valid_states = generatePossibleStates(current_state)
          flagOtptimal = False
            
          for st in range(len(valid_states)):
              
              # Evaluate Actual State
              state = valid_states[st]
              print('\n Model Evaluate... Hill Climbling: ', state)
              metaheuristica_acuracia = objetiveFunction(dataBase,state, nsamples[sample])
              aux_val = metaheuristica_acuracia
              print("Mean Accuracy: %0.3f" % (metaheuristica_acuracia))

              # Select Optimal Value
              if aux_val >= optimal_value:
                  optimal_value = aux_val
                  optimal_state = state
                  current_state = state
                  flagOtptimal = True
         
              # Verify Time Elapsed
              end = time.process_time()
              ElapseTime = end-start
              if ElapseTime >= max_time:
                break
                
               
              print('\n estado ótimo local: ', optimal_state, optimal_value)
    print("\n Best Mean Accuracy: %0.3f" % (optimal_value))
    return optimal_state, optimal_value

"""## Simulated Annealing Function"""

def inicial_state(vectorSize):
  n=vectorSize
  lst = list(itertools.product([0, 1], repeat=n))
  inicialState = random_state(lst)
  # Para não acontecer de selecionar um vetor de zeros
  while np.sum(inicialState) == 0:
    inicialState = random_state(lst)
  return inicialState

def random_state(states):
    index = random.randint(0,len(states)-1)
    return states[index]

import math
import random

def change_probability(value,best_value,t):
    p = 1/(math.exp(1)**((best_value-value)/t))
    r = random.uniform(0,1)
    if r < p:
        return True
    else:
        return False

import time

def simulated_annealing(dataBase, max_time, inter_max, t, alfa):
    # Define n_samples
    nsamples = [3,5,7]

    # Define Process time
    start = time.process_time()
    end = time.process_time()
    ElapseTime = end-start

    # define the model
    model = HP.HeterogeneousClassifier()
    
    # Inicial Values
    optimal_value = 0
    optimal_state = []

    # Evaluate All n_samples
    for sample in range(len(nsamples)):
        valid_states = [1]
        estimatorsVectorSize = nsamples[sample]*3

        # Create inicial State - Criar Randomico
        current_state = inicial_state(estimatorsVectorSize)

        # Evaluate the model for inicial state
        metaheuristica_acuracia = objetiveFunction(dataBase,current_state, nsamples[sample])
        aux_val = metaheuristica_acuracia
        print("Mean Accuracy: %0.3f" % (metaheuristica_acuracia))
        if aux_val > optimal_value:
            optimal_value = aux_val
            optimal_state = current_state
        
        t1 = t
        while t1 >= 1 and end-start <= max_time:
          if len(valid_states) == 0:
              break

          for inter in range(inter_max):    
              valid_states = generatePossibleStates(current_state)
              if len(valid_states) == 0:
                break

              # Evaluate Actual State
              state = random_state(valid_states)
              metaheuristica_acuracia = objetiveFunction(dataBase,state, nsamples[sample])
              aux_val = metaheuristica_acuracia
              print("Mean Accuracy: %0.3f" % (metaheuristica_acuracia))

              # Select Optimal Value
              if aux_val > optimal_value:
                  optimal_value = aux_val
                  optimal_state = state
                  current_state = state
              else:
                  if change_probability(aux_val,optimal_value,t):
                    optimal_value = aux_val
                    optimal_state = state
                    current_state = state
              
              print('\n estado ótimo local: ', optimal_state, optimal_value)
          t1 = t1*alfa
          end = time.process_time()      
               
    print("\n Best Mean Accuracy: %0.3f" % (optimal_value))
    return optimal_state, optimal_value

"""## Genetic Algorithm Function"""

# selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(dataBase, objetiveFunction, n_samples, n_iter, n_pop, r_cross, r_mut):
  max_time = 1800
  start = time.process_time()
  end = time.process_time()
  ElapseTime = end-start
  # Inicial State solution
  pop = [randint(0, 2, n_samples[0]*3).tolist() for _ in range(n_pop)]
  print('Evaluate Inicial state: '+str(pop[0]))
  best, best_eval = 0, objetiveFunction(dataBase, pop[0], n_samples[0])
  print('Inicial state: ' +str(pop[0])+' = '+str(best_eval))
  
  # Evaluate All n_samples
  for sample in range(len(n_samples)):
    # mutation rate
    r_mut = 1.0 / float(n_samples[sample])
    # initial population of random bitstring
    pop = [randint(0, 2, n_samples[sample]*3).tolist() for _ in range(n_pop)]
    # enumerate generations
    for gen in range(n_iter):
      # evaluate all candidates in the population
      print('n_samples: '+str(n_samples[sample])+' - Generation: '+str(gen)+' - Evaluating all candidates...')
      scores = [objetiveFunction(dataBase,c, n_samples[sample]) for c in pop]
      # check for new best solution
      for i in range(n_pop):
        if scores[i] > best_eval:
          best, best_eval = pop[i], scores[i]
          print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
      # select parents
      selected = [selection(pop, scores) for _ in range(n_pop)]
      # create the next generation
      children = list()
      for i in range(0, n_pop, 2):
        # get selected parents in pairs
        p1, p2 = selected[i], selected[i+1]
        # crossover and mutation
        for c in crossover(p1, p2, r_cross):
          # mutation
          mutation(c, r_mut)
          # store for next generation
          children.append(c)
      # replace population
      pop = children
      # Verify Time Elapsed
      end = time.process_time()
      ElapseTime = end-start
  return [best, best_eval]