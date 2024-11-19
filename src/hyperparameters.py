# We have a series of getters and setters in this module.
# You should retrieve hyperparameters using the getters only.
#
# We avoid using constants here so we can adjust the hyperparameters
# dynamically during runtime.
#
# This is helpful when running experiments on the models and comparing
# different combinations of hyperparameters.

lemmatise = False
stem = False
stop_words = False
tf_idf = False
n_gram_length = 2

def set_tfidf(value):
    global tf_idf
    tf_idf = value

def get_tfidf():
    return tf_idf

def set_stop_words(value):
    global stop_words
    stop_words = value

def get_stop_words():
    return stop_words

def set_stemming(value):
    global stem
    stem = value

def get_stemming():
    return stem

def set_lemmatise(value):
    global lemmatise
    lemmatise = value

def get_lemmatise():
    return lemmatise

def set_n_grams(n):
    global n_gram_length
    n_gram_length = n

def get_n_gram():
    return n_gram_length