import matplotlib.pyplot as plt
import numpy as np

from experiments.models import model_names
from experiments.toggle_hyperparameter import toggle_hyperparameter

from src.hyperparameters import set_stemming, set_lemmatise

both_accuracy = toggle_hyperparameter([set_stemming, set_lemmatise])
lemmatisation_accuracy = toggle_hyperparameter([set_lemmatise])
stemming_accuracy = toggle_hyperparameter([set_stemming])

# Plotting grouped bar chart
bar_width = 0.1
index = np.arange(len(model_names))

plt.bar(index, both_accuracy['With'], width=bar_width, label='With Stemming and Lemmatisation')
plt.bar(index + bar_width, lemmatisation_accuracy['With'], width=bar_width, label='With Just Lemmatisation')
plt.bar(index + (2 * bar_width), stemming_accuracy['With'], width=bar_width, label='With Just Stemming')
plt.bar(index + (3 * bar_width), both_accuracy['Without'], width=bar_width, label='Without Stemming and Lemmatisation')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')

plt.title('Model Accuracy with and without Both Stemming and Lemmatisation Applied')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0.81, 0.85)
plt.show()