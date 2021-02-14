"""
Usefulness Evaluation

Uses knn to first predict the class of the given instance using the synthetic exemplars and counter exemplars
and secondly using real sentences from the train set of the respective dataset.
"""

import pickle
import random

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

import vectorize
from pre_processing import get_text_data


# Function to use knn on synthetic exemplars and counter exemplars
def knn_predict_on_synthetic_sentences(instance, number_exemplars, prediction, exemplars_holder,
                                       counter_exemplars_holder):
    knn_X_train = list()
    knn_y_train = list()
    for t in range(len(exemplars_holder + counter_exemplars_holder)):
        knn_X_train = exemplars_holder + counter_exemplars_holder
        if prediction == 0:
            knn_y_train = [0 for _ in range(number_exemplars)] + [1 for _ in range(number_exemplars)]
        else:
            knn_y_train = [1 for _ in range(number_exemplars)] + [0 for _ in range(number_exemplars)]

    knn_X_train, knn_X_test = vectorize.createTFIDF(knn_X_train, instance, remove_stopwords=True, lemmatize=True,
                                                    stemmer=False)

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='brute', leaf_size=30,
                                                 p=2, metric='cosine', metric_params=None, n_jobs=-1)

    knn.fit(knn_X_train, knn_y_train)
    prediction = knn.predict(knn_X_test)
    return prediction[0]


# Function to use knn with real sentences as training data
def knn_predict_on_real_sentences(instance, number_exemplars, sentences, classes):
    np_classes = np.array(classes)
    neutral_sentence_list = list()
    hate_sentence_list = list()
    neutral_prediction_list = list()
    hate_prediction_list = list()

    neutral_class_idx = np.where(np_classes == 0)[0]
    hate_class_idx = np.where(np_classes == 1)[0]

    for t in neutral_class_idx:
        neutral_sentence_list.append(sentences[t])
        neutral_prediction_list.append(classes[t])

    for t in hate_class_idx:
        hate_sentence_list.append(sentences[t])
        hate_prediction_list.append(classes[t])

    neutral_dict = list(zip(neutral_sentence_list, neutral_prediction_list))  # make pairs out of the two lists
    neutral_pairs = random.sample(neutral_dict, number_exemplars)  # pick k random pairs
    X_neutral, y_neutral = zip(*neutral_pairs)  # separate the pairs

    hate_dict = list(zip(hate_sentence_list, hate_prediction_list))  # make pairs out of the two lists
    hate_pairs = random.sample(hate_dict, number_exemplars)  # pick k random pairs
    X_hate, y_hate = zip(*hate_pairs)  # separate the pairs

    X_train = X_neutral + X_hate
    y_train = y_neutral + y_hate

    X_train, X_test = vectorize.createTFIDF(X_train, instance, remove_stopwords=True, lemmatize=True, stemmer=False)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='brute', leaf_size=30,
                                                 p=2, metric='cosine', metric_params=None, n_jobs=-1)

    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    return prediction[0]


# Set dataset, model and maximum nbr of exemplars
dataset = "hate"
model = "RF"
max_nbr_exemplars = 5  # Should not be longer than the max exemplars used for training x-spells

# Load saved exemplars and counter-exemplars for demonstration
with open('../data/' + dataset + '_' + model + '_' + 'exemplars', 'rb') as f:
    loaded_exemplars = pickle.load(f)

with open('../data/' + dataset + '_' + model + '_' + 'counter_exemplars', 'rb') as f:
    loaded_counter_exemplars = pickle.load(f)

X_train, X_test, y_train, y_test, _ = get_text_data('../data/' + dataset + '_tweets.csv', dataset)

# Iterate from 1 to 5 exemplars, inclusive
for no_exemplars in range(1, max_nbr_exemplars + 1):
    knnpreds = list()
    knn_real_sentences_preds = list()
    true_classes = list()

    for j in range(len(loaded_exemplars)):
        true_classes.append(y_test[j])
        exemplars = loaded_exemplars[j][:no_exemplars]
        counter_exemplars = loaded_counter_exemplars[j][:no_exemplars]

        # Train the knn classifier by using synthetic sentences i.e. the exemplars and counter exemplars
        knn_synthetic_prediction = knn_predict_on_synthetic_sentences(X_test[j], no_exemplars, y_test[j], exemplars,
                                                                      counter_exemplars)
        # Train the knn classifier by using real sentences from the train set
        knn_real_sentences_prediction = knn_predict_on_real_sentences(X_test[j], no_exemplars, X_train, y_train)

        knnpreds.append(knn_synthetic_prediction)
        knn_real_sentences_preds.append(knn_real_sentences_prediction)

    knn_synthetic_vs_true_fidelity = accuracy_score(knnpreds, true_classes)
    knn_real_vs_true_fidelity = accuracy_score(knn_real_sentences_preds, true_classes)

    # Print the results of the experiments
    print("number of exemplars: ", no_exemplars)
    print("synthetic knn fidelity is: ", knn_synthetic_vs_true_fidelity)
    print("real knn fidelity is: ", knn_real_vs_true_fidelity)
    print("sample size: ", len(knnpreds))
