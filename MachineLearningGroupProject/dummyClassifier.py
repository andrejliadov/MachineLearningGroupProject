import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, accuracy_score, precision_score
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.dummy import DummyClassifier

#Read in the data
DATA_DIR = "clean-10-categories-data"
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




#Train and test the optimal model
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

vectoriser = TfidfVectorizer(use_idf=False, stop_words='english', max_features=10000, decode_error='ignore', sublinear_tf=True, ngram_range=(1,2))
vectoriser.fit(X_train)

X = vectoriser.transform(data.data)
Y = data.target

model = DummyClassifier(strategy="most_frequent")
model.fit(X, Y)

y_pred = model.predict(vectoriser.transform(X_test))
print(accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

#baseline model
# baselineModel = LogisticRegression()
# baselineModel.fit(vectoriser.transform(X_train), Y_train)

# y_pred = baselineModel.predict(vectoriser.transform(X_test))
# print(accuracy_score(Y_test, y_pred))
# print(classification_report(Y_test, y_pred))
