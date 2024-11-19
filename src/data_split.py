from sklearn.model_selection import train_test_split
import os

def load_text_from_files(directory):
    texts = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            texts.append(text)
    return texts

positive_reviews = load_text_from_files('src/data/pos')
negative_reviews = load_text_from_files('src/data/neg')
all_reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

x_train, x_temp, y_train, y_temp = train_test_split(all_reviews, labels, test_size=0.2, random_state=100)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=100)