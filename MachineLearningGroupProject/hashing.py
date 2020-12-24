import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection._validation import cross_validate
from sklearn.svm._classes import SVC

#Read in the data
DATA_DIR = "../10-categories-data"
data = load_files(DATA_DIR, encoding='utf-8', decode_error='replace')
labels, counts = np.unique(data.target, return_counts=True)
labels_str = np.array(data.target_names)[labels]
print(dict(zip(labels_str, counts)))

#Tokenise and lemmatise the text data
#nltk.download('wordnet')
lemmatiser = WordNetLemmatizer()
tokeniser = CountVectorizer().build_tokenizer()
for i in range(0, len(data.data)):
    temp_str = " "
    data.data[i] = data.data[i].lower()
    data.data[i] = tokeniser(data.data[i])
    for token in range(0, len(data.data[i])):
        data.data[i][token] = lemmatiser.lemmatize(data.data[i][token])
    data.data[i] = temp_str.join(data.data[i])


#Cross-validation analysis of the model
vectoriser = HashingVectorizer(norm=None, stop_words='english', n_features=10000, decode_error='ignore')
vectoriser.fit_transform(data.data)

X = vectoriser.transform(data.data)
Y = data.target
mean_error=[]; std_error=[]
c_range = [0.001, 0.01, 0.1, 0.5, 0.75, 1.0, 5.0]
precision_score = []
precision_var = []

for c in c_range:
    svm_model = SVC(C=c, kernel='rbf', gamma=1)
    scores = cross_validate(svm_model, X, Y, cv=4, scoring=('balanced_accuracy', 'precision_micro', 'f1_micro'))
    precision_score.append(scores['test_precision_micro'].mean())
    precision_var.append(scores['test_precision_micro'].var())
    print('C = %0.3f' % c)
    print('Accuracy: [%s]' % ', '.join(map(str, scores['test_balanced_accuracy'])))
    print('Precision: [%s]' % ', '.join(map(str, scores['test_precision_micro'])))
    print('F1: [%s]' % ', '.join(map(str, scores['test_f1_micro'])))
    print('')

fig = plt.figure()
plt.plot(c_range, precision_score)
plt.errorbar(c_range, precision_score, yerr=precision_var)
plt.title('Precision Vs C')
plt.xlabel('C')
plt.xlim((0,2))
plt.ylabel('Precision')
plt.show()

gamma_range = [0.01, 0.1, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]
precision_score = []
precision_var = []

for g in gamma_range:
    svm_model = SVC(C=0.750, kernel='rbf', gamma=g)
    scores = cross_validate(svm_model, X, Y, cv=4, scoring=('balanced_accuracy', 'precision_micro', 'f1_micro'))
    precision_score.append(scores['test_precision_micro'].mean())
    precision_var.append(scores['test_precision_micro'].var())
    print('Gamma = %0.2f' % g)
    print('Accuracy: [%s]' % ', '.join(map(str, scores['test_balanced_accuracy'])))
    print('Precision: [%s]' % ', '.join(map(str, scores['test_precision_micro'])))
    print('F1: [%s]' % ', '.join(map(str, scores['test_f1_micro'])))
    print('')

fig = plt.figure()
plt.plot(gamma_range, precision_score)
plt.errorbar(gamma_range, precision_score, yerr=precision_var)
plt.title('Precision Vs Gamma')
plt.xlabel('Gamma')
#plt.xlim((0,2))
plt.ylabel('Precision')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

vectoriser = HashingVectorizer(norm=None, stop_words='english', n_features=10000, decode_error='ignore')
vectoriser.fit_transform(data.data)

model = LinearSVC()
model.fit(vectoriser.transform(X_train), Y_train)

y_pred = model.predict(vectoriser.transform(X_test))
print(accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
