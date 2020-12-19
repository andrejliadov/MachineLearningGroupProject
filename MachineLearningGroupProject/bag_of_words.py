import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

#Read in the data
DATA_DIR = "./MachineLearningGroupProject/MachineLearningGroupProject/data/"
data = load_files(DATA_DIR, encoding='utf-8', decode_error='replace')
labels, counts = np.unique(data.target, return_counts=True)
labels_str = np.array(data.target_names)[labels]
print(dict(zip(labels_str, counts)))

#Tokenise and lemmatise the text data
nltk.download('wordnet')
lemmatiser = WordNetLemmatizer()
tokeniser = CountVectorizer().build_tokenizer()
for i in range(0, len(data.data)):
    temp_str = " "
    data.data[i] = data.data[i].lower()
    data.data[i] = tokeniser(data.data[i])
    for token in range(0, len(data.data[i])):
        data.data[i][token] = lemmatiser.lemmatize(data.data[i][token])
    data.data[i] = temp_str.join(data.data[i])


X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

vectoriser = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
vectoriser.fit(X_train)

model = LinearSVC()
model.fit(vectoriser.transform(X_train), Y_train)

y_pred = model.predict(vectoriser.transform(X_test))
print(accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))