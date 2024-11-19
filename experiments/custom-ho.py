from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

from src.data_split import x_train, y_train, x_dev, y_dev, x_test, y_test
from experiments.feature_sets import feature_sets, FeatureGenerator
from experiments.aux.evaluate_model import evaluate_model

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [x for x in np.arange(0.1, 10, 0.1)],
    'solver': ['liblinear', 'saga', 'newton-cg'],
    'fit_intercept': [True, False],
    'max_iter': [x for x in np.arange(100, 10000, 100)],
    'tol': [1e-5, 1e-4, 1e-3, 1e-2],
    'class_weight': [None, 'balanced'],
    'random_state': [42],
    'multi_class': ['ovr', 'multinomial'],
    'l1_ratio': [0.1, 0.5, 0.9]  # Only used if penalty='elasticnet'
}

vectorizer : FeatureGenerator = feature_sets[0]

train_vectors = vectorizer.transform(x_train)
dev_vectors = vectorizer.transform(x_dev)
test_vectors = vectorizer.transform(x_test)

def optimise(hyperparameter):
    accuracies = []
    for parameter in param_grid[hyperparameter]:
        model = LogisticRegression(hyperparameter=parameter)
        accuracies.append(evaluate_model(
            model, train_vectors, y_train, dev_vectors, y_dev
        ))
    return accuracies

accuracies = optimise('solver') # liblinear
accuracies = optimise('penalty') # L1
accuracies = optimise('C') # 0.8

# Generate final test
accuracies = []
tests = []

model = LogisticRegression()
accuracies.append(evaluate_model(model, train_vectors, y_train, dev_vectors, y_dev))
tests.append(evaluate_model(model, train_vectors, y_train, test_vectors, y_test))

model = LogisticRegression(
    solver = 'liblinear',
    C = 0.8,
    penalty = 'l1'
)
accuracies.append(evaluate_model(model, train_vectors, y_train, dev_vectors, y_dev))
tests.append(evaluate_model(model, train_vectors, y_train, test_vectors, y_test))

print(accuracies)
print(tests)

plt.bar(['Default', 'Optimised'], tests)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0.83, 0.85)
plt.title('Accuracy of the Test Set compared to Hyperparameter Optimised')
plt.show()