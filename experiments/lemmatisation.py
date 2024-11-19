import matplotlib.pyplot as plt
import numpy as np

from experiments.aux.models import model_names
from experiments.aux.toggle_hyperparameter import toggle_hyperparameter

from src.hyperparameters import set_lemmatise

accuracy_comparison = toggle_hyperparameter([set_lemmatise])

# Plotting grouped bar chart
bar_width = 0.35
index = np.arange(len(model_names))

plt.bar(index, accuracy_comparison['With Lemmatization'], width=bar_width, label='With Lemmatization')
plt.bar(index + bar_width, accuracy_comparison['Without Lemmatization'], width=bar_width, label='Without Lemmatization')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')

plt.title('Model Accuracy with and without Lemmatization')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0.82, 0.85)
plt.show()