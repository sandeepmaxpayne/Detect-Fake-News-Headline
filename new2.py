import nltk
from nltk import TweetTokenizer
from nltk.corpus import movie_reviews
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sb
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

import random


p = [{"id":1,"category":"crime","headline":"There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV","short_description":"She left her husband. He killed their children. Just another day in America."},{"id":2,"category":"entertainment","headline":"Will Smith Joins Diplo And Nicky Jam For The 2018 World Cup's Official Song","short_description":"Of course it has a song."},{"id":3,"category":"entertainment","headline":"Hugh Grant Marries For The First Time At Age 57","short_description":"The actor and his longtime girlfriend Anna Eberstein tied the knot in a civil ceremony."},{"id":4,"category":"entertainment","headline":"Jim Carrey Blasts 'Castrato' Adam Schiff And Democrats In New Artwork","short_description":"The actor gives Dems an ass-kicking for not fighting hard enough against Donald Trump."},{"id":5,"category":"entertainment","headline":"Julianna Margulies Uses Donald Trump Poop Bags To Pick Up After Her Dog","short_description":"The \"Dietland\" actress said using the bags is a \"really cathartic, therapeutic moment.\""},{"id":6,"category":"entertainment","headline":"Morgan Freeman 'Devastated' That Sexual Harassment Claims Could Undermine Legacy","short_description":"\"It is not right to equate horrific incidents of sexual assault with misplaced compliments or humor,\" he said in a statement."}]
q=[]
r=[]
'''
for i in range(len(p)):
   # print(p[i]['headline'])
    q.append(p[i]['category'])
    r.append(p[i]['headline'])
    text = p[i]['headline']
    tkznr = TweetTokenizer()
    print(tkznr.tokenize(text))
'''
for i in range(len(p)):
    category_map = p[i]
    print(category_map)

    training_data = fetch_20newsgroups(subset='train',
                                       categories=category_map.keys(), shuffle=True, random_state=5)
    count_vectorizer = CountVectorizer()
    train_tc = count_vectorizer.fit_transform(training_data.data)
    print("\nDimensions of training data:", train_tc.shape)


def extract_features(words):
    return dict([(word, True) for word in words])


if __name__ == '__main__':
    # Load the reviews from the corpus
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')

    # Extract the features from the reviews
    features_pos = [(extract_features(movie_reviews.words(
        fileids=[f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(
        fileids=[f])), 'Negative') for f in fileids_neg]

    # Define the train and test split (80% and 20%)
    threshold = 0.8
    num_pos = int(threshold * len(features_pos))
    num_neg = int(threshold * len(features_neg))

    # Create training and training datasets
    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]

    # Print the number of datapoints used
    print('\nNumber of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))