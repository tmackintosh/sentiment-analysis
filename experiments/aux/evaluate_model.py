from sklearn.metrics import accuracy_score

from src.feature_generation import FeatureGenerator
from src.data_split import x_train, x_dev, y_train, y_dev, x_test, y_test

def evaluate_model(model, train_vectors, train_labels, test_vectors, test_labels):
    model.fit(train_vectors, train_labels)
    predictions = model.predict(test_vectors)

    return accuracy_score(predictions, test_labels)

def evaluate_set(set : FeatureGenerator, model, test = False):
    test_set = x_dev
    test_y = y_dev

    if test:
        test_set = x_test
        test_y = y_test

    training_vectors = set.transform(x_train)
    testing_vectors = set.transform(test_set)

    return evaluate_model(model, training_vectors, y_train, testing_vectors, test_y)