# Metaheuristics-Ensemble-Classifiers
*André Paulo Ferreira Machado*

*Metaheuristics for building Ensemble Classifiers*

This work aims to carry out an experimental comparison between a pre-defined set of
learning for automatic classification, based on the idea of using metaheuristics to construct
combinations of classifiers, applied to some classification problems.
The metaheuristics used were: Hill Climbing, Simulated Annealing and Genetic Algorithm.

DataBases: digits, wine e breast cancer.

The results of each classifier are presented in a table containing the average of the accuracies
obtained in each fold of the external cycle, the standard deviation and the confidence interval at 95% of significance
of the results, and also through the boxplot of the results of each classifier in each fold.

The data used in the training set in each test run are standardized (normalized or z-score).
The standardization values obtained from the training data are used to standardize the data of the respective test set.
The experimental procedure of training, validation and testing is carried out through 3 rounds of nested cycles
validation and testing cycle, with the internal validation cycle containing 4 folds and the external testing cycle containing 10 folds.
The grid search of the internal cycle considers the values of n_samples [3,5,7].
