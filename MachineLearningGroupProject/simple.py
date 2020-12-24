import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, accuracy_score, precision_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics._classification import precision_recall_fscore_support
from sklearn.preprocessing._label import LabelBinarizer
from sklearn.metrics._ranking import auc, roc_curve

def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df

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

#Cross-validate The Kernelised SVC Model
vectoriser = TfidfVectorizer(stop_words='english', max_features=10000, decode_error='ignore', sublinear_tf=True, ngram_range=(1,2))
vectoriser.fit(data.data)

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
plt.xlim((0,2))
plt.ylabel('Precision')
plt.show()

#Train and test the optimal model
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

vectoriser = TfidfVectorizer(stop_words='english', max_features=10000, decode_error='ignore', sublinear_tf=True, ngram_range=(1,2))
vectoriser.fit(X_train)

model = SVC(C=0.75, kernel='rbf', gamma=0.75, probability=True)
model.fit(vectoriser.transform(X_train), Y_train)

y_pred_train = model.predict(vectoriser.transform(X_train))
y_pred_test = model.predict(vectoriser.transform(X_test))


report_with_auc_test = class_report(
    y_true=Y_test, 
    y_pred=y_pred_test, 
    y_score=model.predict_proba(vectoriser.transform(X_test)))

print(report_with_auc_test)

report_with_auc_train = class_report(
    y_true=Y_train, 
    y_pred=y_pred_train, 
    y_score=model.predict_proba(vectoriser.transform(X_train)))

print(report_with_auc_train)

print('AUC_macro train: %0.3f' % roc_auc_score(Y_train, y_pred_train), multi_class="oro", average="macro")
print('AUC_macro test: %0.3f' % roc_auc_score(Y_test, y_pred_test), multi_class="oro", average="macro")
print('')
print('Precision_macro train: %0.3f' % precision_score(Y_train, y_pred_train), average='macro')
print('Precision_macro test: %0.3f' % precision_score(Y_test, y_pred_test), average='macro')
print('')

#Baseline model Cross validation
vectoriser = TfidfVectorizer(stop_words='english', max_features=10000, decode_error='ignore', sublinear_tf=True, ngram_range=(1,2))
vectoriser.fit(data.data)

X = vectoriser.transform(data.data)
Y = data.target
mean_error=[]; std_error=[]
c_range = [0.001, 0.01, 0.1, 0.5, 0.75, 1.0, 5.0, 50]
precision_score = []
precision_var = []

for c in c_range:
    svm_model = LogisticRegression(C=c)
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

#baseline model testing
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

vectoriser = TfidfVectorizer(stop_words='english', max_features=10000, decode_error='ignore', sublinear_tf=True, ngram_range=(1,2))
vectoriser.fit(X_train)

baselineModel = LogisticRegression(C=0.75)
baselineModel.fit(vectoriser.transform(X_train), Y_train)

y_pred_train = baselineModel.predict(vectoriser.transform(X_train))
y_pred_test = baselineModel.predict(vectoriser.transform(X_test))

report_with_auc_test = class_report(
    y_true=Y_test, 
    y_pred=y_pred_test, 
    y_score=baselineModel.predict_proba(vectoriser.transform(X_test)))

print(report_with_auc_test)
print('')

report_with_auc_train = class_report(
    y_true=Y_train, 
    y_pred=y_pred_train, 
    y_score=baselineModel.predict_proba(vectoriser.transform(X_train)))

print(report_with_auc_train)