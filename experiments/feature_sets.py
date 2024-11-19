from src.hyperparameters import set_n_grams, set_lemmatise, set_stemming, set_stop_words
from src.feature_generation import FeatureGenerator
from src.tf_idf import TF_IDF
from src.data_split import x_train

feature_sets = []

set_n_grams(3)
set_stop_words(True)

set_lemmatise(True)
set_stemming(False)

vectorizer = FeatureGenerator()
vectorizer.fit(x_train)

feature_sets.append(vectorizer)

set_lemmatise(False)
set_stemming(True)

vectorizer = FeatureGenerator()
vectorizer.fit(x_train)

feature_sets.append(vectorizer)

set_stemming(False)

vectorizer = TF_IDF()
vectorizer.fit(x_train)

feature_sets.append(vectorizer)