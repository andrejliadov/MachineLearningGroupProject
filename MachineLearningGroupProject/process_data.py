import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from sklearn.feature_extraction.text import TfidfVectorizer

#Parse words in the text files into their catagories
def parse_data(catagory):
    root_dir = 'data/'
    text_data = ""
    text_data_list = []

    for filename in glob.iglob(root_dir + catagory + '/*.txt', recursive=True):
        with open(filename, 'r') as file:
            text_data = file.read()
            text_data_list.append(text_data)

    return text_data_list

#Return a feature vector for each document
def get_feature_vector(document_list):

    vectorizer = TfidfVectorizer(norm=None)
    tfidf_vector = vectorizer.fit_transform(document_list)
    print(vectorizer.get_feature_names())

    return tfidf_vector

#The data is parsed here
culture_text_list = parse_data('Culture')
geography_text_list = parse_data('Geography')
physics_text_list = parse_data('Physics')

#Get the feature vectors for documents in every catagory
culture_features = get_feature_vector(culture_text_list)
culture_features = get_feature_vector(geography_text_list)
culture_features = get_feature_vector(physics_text_list)