import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Read in the data
DATA_DIR = "MachineLearningGroupProject/data/"
data = load_files(DATA_DIR, encoding='utf-8', decode_error='replace')
labels, counts = np.unique(data.target, return_counts=True)
labels_str = np.array(data.target_names)[labels]
print(dict(zip(labels_str, counts)))

label_encoder = LabelEncoder()
tokeniser = CountVectorizer().build_tokenizer()
integer_encoder = label_encoder
enc = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
array = []
for i in range(0, len(data.data)):
    data.data[i] = data.data[i].lower()
    data.data[i] = tokeniser(data.data[i])
    integer_encoded = integer_encoder.fit_transform(data.data[i])
    array.append(integer_encoded.reshape(len(integer_encoded), 1))

onehot_encoded = []

for i in range(0, len(array)):
    onehot_encoded.append(enc.fit_transform(array[i]))

