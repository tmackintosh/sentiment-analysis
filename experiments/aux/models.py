from src.naive_bayes import NaiveBayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from experiments.aux.evaluate_model import evaluate_model

models = [NaiveBayes(), MultinomialNB(), LogisticRegression(), svm.SVC()]
model_names = ['Naive Bayes', 'Multinomial NB', 'Logistic Regression', 'SVM']

def get_accuracy(model, train_vectors, test_vectors, train_y, test_y):
    return evaluate_model(model, train_vectors, train_y, test_vectors, test_y)