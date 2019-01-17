import itertools
import json
import numpy as np
import nltk
from sklearn import (
            datasets, feature_extraction, model_selection, pipeline,
            naive_bayes, metrics
)
import matplotlib.pylab as plt

def extract_featuers(corpus):
    stop_words = nltk.corpus.stopwords.words("english")

    count_vectorizer = feature_extraction.text.CountVectorizer(
    lowercase=True,
    tokenizer=nltk.word_tokenize,
    min_df=2,
    ngram_range=(1,2),
    stop_words=stop_words)
    process_corpus = count_vectorizer.fit_transform(corpus)
    process_corpus = feature_extraction.text.TfidfTransformer().fit_transform(process_corpus)
    return process_corpus

if __name__ == '__main__':
    with open('dataset.json', encoding='ISO-8859-1') as f:
        news_data = json.loads(f.read())

    print('Data loaded .\nClasses= {classes}\n{datapoints}'.format(
        classes=news_data,
        datapoints=len(news_data)
    ))
    print(news_data[0])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        news_data, news_data, test_size=0.33,random_state=200004)

    stop_words = nltk.corpus.stopwords.words("english")

    model = pipeline.Pipeline([
        ('counts', feature_extraction.text.CountVectorizer(
            lowercase=True,
            tokenizer=nltk.word_tokenize,
            min_df=2,
            ngram_range=(1,2),
            stop_words=stop_words
        )),
        ('tfjdf', feature_extraction.text.TfidfTransformer()),
        ('naivebayes', naive_bayes.MultinomialNB()),
    ])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('Accuracy of multinomial naive bayes={}'.format(np.mean(y_pred == y_test)))
    print(metrics.classification_report(
        y_test, y_pred, target_names=news_data
    ))

    grid_search_model = model_selection.GridSearchCV(
        model,
        {
            'counts__ngram_range': [(1, 1), (1,2)],
            'naivebayes__alpha':(0.1, 3.0)
        },
        n_jobs=-1
    )
    grid_search_model.fit(x_train, y_train)
    print(grid_search_model.cv_results_)
