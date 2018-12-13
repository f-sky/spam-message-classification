import csv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from spam_utils_v1 import Corpus


def train_classifier(corpus: Corpus, mode):
    textFeatures = corpus.train_sentences.copy() + corpus.test_sentences.copy()
    num_train_and_dev = len(corpus.train_sentences)
    vectorizer = TfidfVectorizer("english")
    features_total = vectorizer.fit_transform(textFeatures)
    features = features_total[0:num_train_and_dev]
    test_features = features_total[num_train_and_dev:]
    features_train, features_test, labels_train, labels_test = train_test_split(features, corpus.labels, test_size=0.3,
                                                                                shuffle=False)
    if mode == 'svm':
        from sklearn.svm import SVC

        model: SVC = SVC(kernel='sigmoid', gamma=1.0)
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        print(accuracy_score(labels_test, prediction))
    elif mode == 'multinomialnb':
        from sklearn.naive_bayes import MultinomialNB

        model = MultinomialNB(alpha=0.2)
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        print(accuracy_score(labels_test, prediction))
    elif mode == 'mlp':
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), verbose=True)
        print(model)
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        print(accuracy_score(labels_test, prediction))
    return model, test_features


def test_classifier(model, test_features, corpus: Corpus):
    predictions = [['SmsId', 'Label']]
    ys = model.predict(test_features)
    ys = list(map(lambda x: 'spam' if x == 1 else 'ham', ys))
    predictions = predictions + np.array([corpus.test_smsids, ys]).transpose().tolist()
    with open('data/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)
