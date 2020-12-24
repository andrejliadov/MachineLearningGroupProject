import numpy as np
# import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Read in the data
DATA_DIR = "./data/"
data = load_files(DATA_DIR, encoding='utf-8', decode_error='replace')
labels, counts = np.unique(data.target, return_counts=True)
labels_str = np.array(data.target_names)[labels]
print(dict(zip(labels_str, counts)))

#Tokenise and lemmatise the text data
le = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
lemmatiser = WordNetLemmatizer()
tokeniser = CountVectorizer().build_tokenizer()
for i in range(0, len(data.data)):
    temp_str = " "
    data.data[i] = data.data[i].lower()
    data.data[i] = tokeniser(data.data[i])
    for token in range(0, len(data.data[i])):
        data.data[i][token] = lemmatiser.lemmatize(data.data[i][token])
    data.data[i] = temp_str.join(data.data[i])

label_encoder = le.fit_transform(data.data)
integer_encoder = label_encoder.reshape(len(label_encoder), 1)
onehot_encoder = onehot_encoder.fit_transform(integer_encoder)

X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)
