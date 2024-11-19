import numpy as np
import matplotlib.pyplot as plt

from src.hyperparameters import set_n_grams, get_n_gram
from src.feature_generation import FeatureGenerator
from src.data_split import x_train, x_dev, y_train, y_dev

from experiments.models import models, model_names, get_accuracy

n_values = np.arange(1, 6)

total_accuracies = []
for n in n_values:
    set_n_grams(n)

    print('N-Gram', get_n_gram())
    generator = FeatureGenerator()
    generator.fit(x_train)

    train_vectors = generator.transform(x_train)
    test_vectors = generator.transform(x_dev)

    accuracies = [get_accuracy(model, train_vectors, test_vectors, y_train, y_dev) for model in models]
    
    total_accuracies.append(accuracies)
    
plt.plot(n_values, total_accuracies)
plt.title('Accuracy vs. n for Different Models')
plt.xlabel('n')
plt.ylabel('Accuracy')

plt.xticks(np.arange(min(n_values), max(n_values)+1, 1.0))

plt.legend(model_names)
plt.show()