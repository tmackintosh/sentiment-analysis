import numpy as np
from scipy.sparse import csr_matrix

class NaiveBayes:
    def __init__(self):
        self.alpha = 1
        self.prior_probabilities = None
        self.feature_probs = None

        self.classes = np.array([0, 1])

    def fit(self, vectors : csr_matrix, labels):
        num_features = vectors.shape[1]

        self.prior_probabilities = {
            0 : 1 - np.mean(labels),
            1 : np.mean(labels)
        }

        # Shape of 2, num_features because of the 2 classes,
        # positive and negative.
        self.feature_probs = np.zeros((len(self.classes), num_features))

        for sentiment in self.classes:
            # A vector that acts as an indicator function for whether a
            # document has this sentiment value. Takes the values of 0 and 1
            # for the indication.
            class_mask = (labels == sentiment)

            class_count = np.sum(class_mask)

            # This numpy trick allows us to count how many features appeared
            # in the documents of this class and represent it as a vector
            # for all features.
            class_occurrences = vectors[class_mask].sum(axis=0)

            # Laplace smoothing
            class_occurrences += self.alpha
            class_count += (len(self.classes) * self.alpha)

            # As class_occurences is a vector of all features, we can
            # simulatenously calculate the feature probabilities for
            # all features per class.
            self.feature_probs[sentiment] = class_occurrences / class_count

    def predict(self, vectors : csr_matrix):
        num_documents = vectors.shape[0]
        predictions = np.zeros(num_documents)

        for i in range(num_documents):
            document_likelihoods = np.zeros(len(self.classes))

            for sentiment in self.classes:
                # Bayes theorem
                likelihood = np.sum(np.log(self.feature_probs[sentiment, :]) * vectors[i, :].toarray())
                likelihood += np.log(self.prior_probabilities[sentiment])
                document_likelihoods[sentiment] = likelihood

            predictions[i] = self.classes[np.argmax(document_likelihoods)]

        return predictions