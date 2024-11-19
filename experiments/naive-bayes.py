from src.naive_bayes import NaiveBayes
from experiments.feature_sets import feature_sets
from src.data_split import x_train, y_train, x_test, y_test
from src.feature_generation import FeatureGenerator
from experiments.evaluate_model import evaluate_model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

accuracies = {
    'Custom' : [],
    'scikit' : []
}

for set in feature_sets:
    set : FeatureGenerator

    training_vectors = set.transform(x_train)
    testing_vectors = set.transform(x_test)

    accuracies['Custom'].append(evaluate_model(NaiveBayes(), training_vectors, y_train, testing_vectors, y_test))
    accuracies['scikit'].append(evaluate_model(MultinomialNB(), training_vectors, y_train, testing_vectors, y_test))

# Plotting grouped bar chart
bar_width = 0.3
index = np.arange(3)

plt.bar(index, accuracies['Custom'], width=bar_width, label='Our Naive Bayes Model')
plt.bar(index + bar_width, accuracies['scikit'], width=bar_width, label='sklearn\'s MultinomialNB model')
# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, ['Set 1', 'Set 2', 'Set 3'], rotation=45, ha='right')

plt.title('Model Accuracy of the Naive Bayes Model for All Feature Sets')
plt.xlabel('Feature Sets')
plt.ylabel('Accuracy')
plt.ylim(0.82, 0.87)
plt.legend()
plt.show()