import matplotlib.pyplot as plt
import numpy as np

from experiments.models import model_names
from experiments.toggle_hyperparameter import toggle_hyperparameter

from src.hyperparameters import set_tfidf

accuracy_comparison = toggle_hyperparameter([set_tfidf])

# Plotting grouped bar chart
bar_width = 0.35
index = np.arange(len(model_names))

plt.bar(index, accuracy_comparison['With'], width=bar_width, label='With TF-IDF Value Representation')
plt.bar(index + bar_width, accuracy_comparison['Without'], width=bar_width, label='With Feature Count Representation')

# Rotate x-axis labels if needed
plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')

plt.title('Model Accuracy with TF IDF Values against Feature Counts')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0.75, 0.88)
plt.show()