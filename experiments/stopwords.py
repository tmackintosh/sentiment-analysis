import matplotlib.pyplot as plt
import numpy as np

from experiments.models import model_names
from experiments.toggle_hyperparameter import toggle_hyperparameter

from src.hyperparameters import set_stop_words

accuracy_comparison = toggle_hyperparameter([set_stop_words])

# Plotting grouped bar chart
bar_width = 0.35
index = np.arange(len(model_names))

plt.bar(index, accuracy_comparison['With'], width=bar_width, label='With Stop Words Removed')
plt.bar(index + bar_width, accuracy_comparison['Without'], width=bar_width, label='Without Stop Words Removed')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')

plt.title('Model Accuracy with and without the Removal of Stop Words')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0.82, 0.84)
plt.show()