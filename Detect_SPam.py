import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn import datasets, feature_extraction, model_selection, linear_model
from collections import Counter
import csv
import nltk

from pprint import pprint

from detect2 import load_file

p=[]
q=[]
with open('dataset.json', encoding='utf-8') as f:
    data = json.loads(f.read())
    for i in range(len(data)):
        p.append(data[i]['headline'])
        q.append(data[i]['category'])
    #unique_word  = list(set(" ".join(data1).split()))


def extract_features(corpus):
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=nltk.word_tokenize,
        stop_words='english'
        #min_df=1
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
        processed_corpus
    )
    return processed_corpus


data_directory = 'Machine_Learning/dataset'

sentiment_data = datasets.load_files(data_directory, shuffle=True)
print('{} files loaded.'.format(len(sentiment_data.data)))
print('they contain the following headlines: {}'.format(sentiment_data.target_names))

news_tfidf = extract_features(sentiment_data.data)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    news_tfidf, sentiment_data.target, test_size=0.30, random_state=42)

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)
print('Model performance: {}'.format(model.score(x_test, y_test)))

y_pred = model.predict(x_test)
for i in range(5):
    print('Headline:\n{headline}\nCorrect label:{correct}; Predicted:{predict}'.format(
        headline=x_test[i], correct=y_test[i], predict=y_pred[i]
    ))
