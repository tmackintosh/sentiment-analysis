from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_split import x_train, y_train, x_dev, y_dev
from experiments.feature_sets import feature_sets

# Assume you have your data X and y ready
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'fit_intercept': [ True, False ],
    'max_iter': [100, 200, 300, 400],
    'tol': [1e-4, 1e-3, 1e-2],
}

training_vectors = feature_sets[0].transform(x_train)
test_vectors = feature_sets[0].transform(x_dev)

logreg = LogisticRegression()

grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=2)

grid_search.fit(training_vectors, y_train)

best_params = grid_search.best_params_

best_logreg = grid_search.best_estimator_

y_pred = best_logreg.predict(test_vectors)

best_params = grid_search.best_params_
accuracy = accuracy_score(y_dev, y_pred)