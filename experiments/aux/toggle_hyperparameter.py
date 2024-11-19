from src.feature_generation import FeatureGenerator
from src.tf_idf import TF_IDF
from src.data_split import x_train, x_dev, y_train, y_dev
from src.hyperparameters import get_tfidf

from experiments.aux.models import get_accuracy, models

def toggle_hyperparameter(setters):
    # Initialise accuracies as an array of length 2 to reflect the 2
    # boolean values that will populate it. As this is a toggler,
    # we are only ever toggling between True and False.
    accuracies = [None, None]

    for value in [True, False]:
        for setter in setters:
            setter(value)

        if get_tfidf():
            generator = TF_IDF()
        else:
            generator = FeatureGenerator()

        generator.fit(x_train)

        train_vectors = generator.transform(x_train)
        test_vectors = generator.transform(x_dev)

        accuracies[int(value)] = [
            get_accuracy(model, train_vectors, test_vectors, y_train, y_dev) 
            for model in models]

    # Use more descriptive variable names
    accuracy_comparison = {
        'With': accuracies[1],
        'Without': accuracies[0]
    }

    return accuracy_comparison