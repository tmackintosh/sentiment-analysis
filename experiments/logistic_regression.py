from experiments.feature_sets import feature_sets
from experiments.aux.evaluate_model import evaluate_set

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

accuracies = [evaluate_set(feature_set, LogisticRegression()) for feature_set in feature_sets]
test_accuracy = evaluate_set(feature_sets[0], LogisticRegression(), test = True)

print(test_accuracy)

# Plotting grouped bar chart
bar_width = 0.5
index = np.arange(3)

plt.bar(index, accuracies, width=bar_width, label='Logistic Regression Model')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, ['Set 1', 'Set 2', 'Set 3'], rotation=45, ha='right')

plt.title('Model Accuracy of the Logistic Regression Model for All Feature Sets')
plt.xlabel('Feature Sets')
plt.ylabel('Accuracy')
plt.ylim(0.775, 0.825)
plt.legend()
plt.show()