from sklearn import svm
from sklearn.svm import NuSVC, LinearSVC

import matplotlib.pyplot as plt
import numpy as np

from experiments.aux.evaluate_model import evaluate_model
from experiments.feature_sets import feature_sets
from src.data_split import x_train, y_train, x_dev, y_dev, x_test, y_test

train_vectors = feature_sets[0].transform(x_train)
dev_vectors = feature_sets[0].transform(x_dev)
test_vectors = feature_sets[0].transform(x_test)

param_grid = {
    'models' : [svm.SVC(), NuSVC(), LinearSVC()],
    'C' : [x for x in np.arange(0.1, 10, 0.1)],
    'dual' : [True, False],
    'loss' : ['hinge', 'square_hinge'],
    'tol' : [1 * (10 ^ -x) for x in np.arange(100, 10000, 100)],
    'max_iter' : [x for x in np.arange(100, 10000, 100)],
    'fit_transform' : [True, False]
}

def optimise(hyperparameter):
    accuracies = []
    for parameter in param_grid[hyperparameter]:
        model = LinearSVC(C = 3, dual = False, loss = parameter)
        accuracies.append(evaluate_model(
            model, train_vectors, y_train, dev_vectors, y_dev
        ))
    return accuracies

accuracies = optimise('models')
accuracies = optimise('C')
accuracies = optimise('dual')
accuracies = optimise('loss')
accuracies = optimise('tol')
accuracies = optimise('max_iter')
accuracies = optimise('fit_transform')

# print(accuracies)

accuracies = [0.8075, evaluate_model(LinearSVC(C = 3, dual = True, loss = 'hinge'),
                                     train_vectors, y_train, test_vectors, y_test)]

print(accuracies[1])

plt.bar(['Default', 'Optimised'], accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracies')

plt.ylim(0.80, 0.84)
plt.legend()
plt.title('Comparison in accuracy between the default hyperparameters and optimised ones.')
plt.show()