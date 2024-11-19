import matplotlib.pyplot as plt

accuracies = [0.8875, 0.8975]
models = ['Uncased', 'Cased']

plt.bar(models, accuracies)

plt.ylim(0.88, 0.9)

plt.xlabel('BERT Flavour')
plt.ylabel('Accuracy')

plt.title('Accuracies of each flavour of BERT on the test data')
plt.show()