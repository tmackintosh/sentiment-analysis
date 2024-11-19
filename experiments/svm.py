from experiments.feature_sets import feature_sets
from experiments.aux.evaluate_model import evaluate_set

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm

accuracies = [evaluate_set(feature_set, svm.SVC()) for feature_set in feature_sets]
test_accuracy = evaluate_set(feature_sets[0], svm.SVC(), test = True)

print(test_accuracy)

# Plotting grouped bar chart
bar_width = 0.5
index = np.arange(3)

plt.bar(index, accuracies, width=bar_width, label='SVM Model')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, ['Set 1', 'Set 2', 'Set 3'], rotation=45, ha='right')

plt.title('Model Accuracy of the SVM Model for All Feature Sets')
plt.xlabel('Feature Sets')
plt.ylabel('Accuracy')
plt.ylim(0.79, 0.81)
plt.legend()
plt.show()