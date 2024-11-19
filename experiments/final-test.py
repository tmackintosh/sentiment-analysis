from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

import numpy as np

from src.naive_bayes import NaiveBayes
from src.data_split import x_train, x_test, y_train, y_test

from experiments.feature_sets import feature_sets
from experiments.aux.evaluate_model import evaluate_model

naive_bayes = NaiveBayes()
multinomial = MultinomialNB()
logistic = LogisticRegression()
svm = LinearSVC(C = 3, dual = True, loss = 'hinge')

total_accuracies = []

for feature_set in feature_sets:
    training_vectors = feature_set.transform(x_train)
    test_vectors = feature_set.transform(x_test)

    models = [naive_bayes, multinomial, logistic, svm]
    accuracies = [evaluate_model(model, training_vectors, y_train, test_vectors, y_test) for model in models]

    accuracies.append(0.8875)
    accuracies.append(0.8975)

    total_accuracies.append(accuracies)

print(total_accuracies)

# Extract the model names for the y-axis
model_names = ['Naive Bayes', 'MultinomialNB', 'LogisticRegression', 'SVM', 'BERT (Uncased)', 'BERT (Cased)']

# Extract the accuracies for each model and feature set
accuracies = np.array(total_accuracies)

# Define the number of models and feature sets
num_models = len(model_names)
num_feature_sets = len(feature_sets)

# Define the width of each bar
bar_width = 0.15

# Set up the y-axis positions for each group of bars
y = np.arange(num_models)

# Create a figure and axis for the bar chart
fig, ax = plt.subplots()

# Create bars for each feature set and model
for i in range(num_feature_sets):
    bar_positions = y + i * bar_width
    ax.barh(bar_positions, accuracies[i, :], height=bar_width, label=f'Feature Set {i+1}')

# Set the y-axis labels to be the model names
ax.set_yticks(y + (bar_width * (num_feature_sets - 1)) / 2)
ax.set_yticklabels(model_names)

# Set labels and title
plt.xlabel('Accuracies')
plt.ylabel('Models')
plt.xlim(0.8, 0.9)
plt.title('Overview of all feature sets\' performance by model')

# Add a legend to distinguish the feature sets
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()