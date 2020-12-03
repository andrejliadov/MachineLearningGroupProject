import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem import PorterStemmer

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
def get_feature_vector(document_stem_list):
    tfidf_vector_list = []

    vectorizer = TfidfVectorizer(norm=None)
    for i in range(0, len(document_stem_list)):
        tfidf_vector_list.append(vectorizer.fit_transform(document_stem_list[i]))
        #print(vectorizer.get_feature_names())

    return tfidf_vector_list

#Download a stop words list and tokenise the data
def tokenize_data(document_list):
    nltk.download('stopwords')
    vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
    tokenizer = vectorizer.build_tokenizer()
    document_tokens_list = []

    for i in range(0, len(document_list)):
        document_tokens_list.append(tokenizer(document_list[i]))

    return document_tokens_list

def stem_data(document_tokens_list):
    stemmer = PorterStemmer()
    document_stems_list = []

    for i in range(0, len(document_tokens_list)):
        stem_list = []
        for token in document_tokens_list[0]:
            stem = stemmer.stem(token)
            stem_list.append(stem)
        document_stems_list.append(stem_list)
    
    return document_stems_list

#The data is parsed here
culture_text_list = parse_data('Culture')
geography_text_list = parse_data('Geography')
physics_text_list = parse_data('Physics')

#Tokenise the data
culture_tokens_list = tokenize_data(culture_text_list)
geography_tokens_list = tokenize_data(geography_text_list)
physics_tokens_list = tokenize_data(physics_text_list)

#Apply a Porter Stemming to the tokens
culture_stems_list = stem_data(culture_tokens_list)
geography_stems_list = stem_data(geography_tokens_list)
physics_stems_list = stem_data(physics_tokens_list)

#Get the feature vectors for documents in every catagory
culture_features = get_feature_vector(culture_stems_list)
culture_features = get_feature_vector(geography_stems_list)
culture_features = get_feature_vector(physics_stems_list)