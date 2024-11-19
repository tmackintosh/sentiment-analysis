from src.feature_generation import FeatureGenerator

import numpy as np
from scipy.sparse import csr_matrix

class TF_IDF():
    def __init__(self):
        self.vectorizer = None
        self.idf = None

    def fit(self, documents):
        self.vectorizer = FeatureGenerator()

        dtm = self.vectorizer.fit_transform(documents)

        df = np.sum(dtm > 0, axis=0)
        self.idf = np.log((len(documents)) / (1 + df))
        self.idf = self.idf.flatten()

    def transform(self, document):
        dtm : csr_matrix = self.vectorizer.transform(document)
        tf = dtm / np.sum(dtm, axis=1)
        tfidf_matrix = csr_matrix(tf.multiply(self.idf))

        return tfidf_matrix