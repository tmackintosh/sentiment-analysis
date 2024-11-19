from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from src.hyperparameters import get_n_gram, get_lemmatise, get_stemming, get_stop_words

nltk.download('stopwords')

# We inherit CountVectorizer class for no theoretical reason,
# instead it allows IDEs to successfully lint the methods
# of our generator without being explicitly defined.
#
# See the __getatr__ for more details.
# This is purely a developer quality of life feature.
class FeatureGenerator(CountVectorizer):
    def __init__(self):
        self.vectorizer = CountVectorizer(
            ngram_range = (1, get_n_gram()),
            preprocessor = self.process_text,
            stop_words = stopwords.words('english') if get_stop_words() else None
        )

    def process_text(self, text):
        tokens = word_tokenize(text)

        if get_lemmatise():
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        if get_stemming():
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

    def __getattr__(self, name):
        # If an attribute is not found, try to access it from self.vectorizer
        # This will allow us to write 'object.fit_transform' instead of
        # 'object.vectorizer.fit_transform'.
        # 
        # This is purely a developer quality of life feature.
        if hasattr(self.vectorizer, name):
            return getattr(self.vectorizer, name)
        raise AttributeError(f"'FeatureGenerator' object has no attribute '{name}'")