import matplotlib.pyplot as plt
import numpy as np

from experiments.models import model_names
from experiments.toggle_hyperparameter import toggle_hyperparameter

from src.hyperparameters import set_stemming

accuracy_comparison = toggle_hyperparameter([set_stemming])

# Plotting grouped bar chart
bar_width = 0.35
index = np.arange(len(model_names))

plt.bar(index, accuracy_comparison['With'], width=bar_width, label='With Stemming')
plt.bar(index + bar_width, accuracy_comparison['Without'], width=bar_width, label='Without Stemming')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')

plt.title('Model Accuracy with and without Stemming')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0.81, 0.85)
plt.show()