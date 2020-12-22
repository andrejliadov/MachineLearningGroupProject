# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import glob

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

#Parse words in the text files into their catagories
def parse_data(catagory):
    root_dir = 'MachineLearningGroupProject/MachineLearningGroupProject/10-categories-data/'
    text_data = ""
    text_data_list = []

    for filename in glob.iglob(root_dir + catagory + '/*.txt', recursive=True):
        with open(filename, 'r') as file:
            text_data = file.read()
            text_data_list.append(text_data)

    return text_data_list

#Combine parsed data into one list per catagory
def combine_lists(text_list):
    text_data = []
    for i in range(0, len(text_list)):
        text_data = text_data + text_list

    return text_data

#Return a feature vector for each document
def get_feature_vector(document_stem_list, vectorizer):
    tfidf_vector_list = []

    for i in range(0, len(document_stem_list)):
        tfidf_vector_list.append(vectorizer.fit_transform(document_stem_list[i]))
        #print(vectorizer.get_feature_names())

    return tfidf_vector_list

#Download a stop words list and tokenise the data
def tokenize_data(text_data):
    nltk.download('stopwords')
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    tokenizer = vectorizer.build_tokenizer()
    
    tokens_list = tokenizer(text_data)

    return tokens_list

#Download a stop words list and tokenise a list of documents
def tokenize_data_list(document_list):
    nltk.download('stopwords')
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    tokenizer = vectorizer.build_tokenizer()
    document_tokens_list = []

    for i in range(0, len(document_list)):
        text = document_list[i]
        text = text.lower()
        tokens = tokenizer(text)
        document_tokens_list.append(tokens)

    return document_tokens_list

#Convert the tokens into stems
def stem_data(document_tokens_list):
    stemmer = PorterStemmer()
    document_stems_list = []

    for i in range(0, len(document_tokens_list)):
        stem_list = []
        for token in document_tokens_list[i]:
            stem = stemmer.stem(token)
            stem_list.append(stem)
        
        document_stems_list.append(stem_list)
    
    return document_stems_list

#Convert the tokens into lemmas
def lemmatise_data(document_tokens_list):
    nltk.download('wordnet')
    lemmatiser = WordNetLemmatizer()
    document_lemmas_list = []

    for i in range(0, len(document_tokens_list)):
        lemma_list = []
        for token in document_tokens_list[i]:
            lemma = lemmatiser.lemmatize(token)
            lemma_list.append(lemma)
        document_lemmas_list.append(lemma_list)

    return document_lemmas_list

#The data is parsed here
culture_text_list = parse_data('Culture')
geography_text_list = parse_data('Geography')
physics_text_list = parse_data('Physics')

#Combine words on a topic
#culture_text_data = combine_lists(culture_text_list)
#geography_text_data = combine_lists(geography_text_list)
#physics_text_data = combine_lists(physics_text_list)

#Tokenise the data
culture_tokens_list = tokenize_data_list(culture_text_list)
geography_tokens_list = tokenize_data_list(geography_text_list)
physics_tokens_list = tokenize_data_list(physics_text_list)

#Apply a Porter Stemming to the tokens
culture_stems_list = lemmatise_data(culture_tokens_list)
geography_stems_list = lemmatise_data(geography_tokens_list)
physics_stems_list = lemmatise_data(physics_tokens_list)

#Get the feature vectors for documents in every catagory
vectorizer = TfidfVectorizer(norm=None, max_features=5000)
culture_features = get_feature_vector(culture_stems_list, vectorizer)
geography_features = get_feature_vector(geography_stems_list, vectorizer)
physics_features = get_feature_vector(physics_stems_list, vectorizer)
