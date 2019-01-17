import json
import collections
from pprint import pprint

import nltk

import os
import random

from sklearn.preprocessing import label

stop_words = {
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again',
    'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
    'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',

}
p=[]
q=[]
r=[]

def load_file():

    with open('dataset.json', encoding='utf-8') as f:
        data = json.loads(f.read())
      #  data1 = pd.read_json('dataset.json')
        for i in range(len(data)):
            q.append(data[i]['category'])
            p.append(data[i]['headline'])
            r.append(data[i]['short_description'])

        return data
load_file()

def preprocess_sentence(sentence):
    lematizer = nltk.word_tokenize()

    processed_tokens = nltk.word_tokenize(sentence)
    processed_tokens = [w.lower() for w in processed_tokens]

    word_counts = collections.Counter(processed_tokens)
    uncommon_words = word_counts.most_common()[:-10:-1]

    processed_tokens = [w for w in processed_tokens if w not in stop_words]
    processed_tokens = [w for w in processed_tokens if w not in uncommon_words]
    processed_tokens = [lematizer.lematize(w) for w in processed_tokens]
    return processed_tokens

def feature_extraction(tokens):
    return dict(collections.Counter(tokens))

def train_test_split(dataset, train_size=0.8):
    num_train = int(len(dataset) * train_size)
    return dataset[:num_train], dataset[num_train:]

positive = load_file()
negative = load_file()

positive = [(p,1) for p in positive]
negative = [(p,0) for p in negative]

all_posneg = positive + negative
random.shuffle(all_posneg)

print('{} headlines processed .'.format(len(all_posneg)))

featurized = [(feature_extraction(corpus), label)
              for corpus, label in all_posneg]
training_set, test_set = train_test_split(featurized, train_size=0.7)

model = nltk.classify.NaiveBayesClassifier.train(training_set)
training_error = nltk.classify.accuracy(model, training_set)
print('Model training complete. Accuracy on training set: {}'.format(training_error))


testing_error = nltk.classify.accuracy(model, test_set)
print('Accuracy on test set: {}'.format(testing_error))



