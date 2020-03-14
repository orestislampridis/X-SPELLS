import pickle
import re

import numpy as np
import pandas as pd
import sklearn
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import stdev
import decision_tree
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial
from collections import Counter


def my_clean(text):
    text = text.lower().split()
    text = [w for w in text]
    text = " ".join(text)
    text = re.sub(r"rt", "", text)
    return text


# Removes 'rt' from all input data
def preProcessing(strings):
    clean_tweet_texts = []
    for string in strings:
        clean_tweet_texts.append(my_clean(string))
    return clean_tweet_texts


def get_text_data(num_samples, data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    # Removing the offensive comments, keeping only neutral and hatespeech,
    # in order to convert the problem to a simple binary classification problem
    df = df[df['class'] != 1]
    X = df['tweet'].values
    y = df['class'].values
    class_names = ['hate', 'offensive', 'neutral']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

    new_X_test = preProcessing(X_test)

    return X_test, y_test, new_X_test


def find_closest_k_sentences(sentences, ids, k, metric):
    index_list = ids
    final_idx_distances = list()
    sentences = [sentences[x] for x in ids]
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(sentences).toarray()

    dictionary = dict(zip(index_list, sentences_vectors))
    print(len(dictionary))
    distances = [[] for _ in range(len(sentences))]
    distances_dict = [dict() for _ in range(len(sentences))]
    idx_distances = list()
    count = 0
    for idx in index_list:
        print('idx: ', idx)
        instance = dictionary.get(idx)
        instance = np.array(instance)

        for j in index_list:
            print('j: ', j)
            temp_state_sentence = dictionary.get(j)

            distances[count].append(cdist(instance.reshape(1, -1), temp_state_sentence.reshape(1, -1), metric=metric).ravel())
            idx_distances.append(j)

        distances_dict[count] = dict(zip(idx_distances, distances[count]))
        distances_sorted = {k: v for k, v in sorted(distances_dict[count].items(), key=lambda x: x[1])}
        final_idxs, final_dists = zip(*list(distances_sorted.items()))
        final_idx_distances.append(final_idxs[1:k + 1])
        count += 1

    return index_list, final_idx_distances


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


res = get_text_data(num_samples=20000, data_path="data/hate_tweets.csv")

X_original, y_original, X_original_processed = res

with open('data/ids', 'rb') as f:
    loaded_ids = pickle.load(f)

with open('data/exemplars', 'rb') as f:
    loaded_exemplars = pickle.load(f)

with open('data/counter_exemplars', 'rb') as f:
    loaded_counter_exemplars = pickle.load(f)

with open('data/top_exemplar_words', 'rb') as f:
    loaded_top_exemplar_words = pickle.load(f)

with open('data/top_counter_exemplar_words', 'rb') as f:
    loaded_top_counter_exemplar_words = pickle.load(f)


'''Find closest k sentences for final experiment'''
closest_k = 10
index, closest_indexes = (find_closest_k_sentences(X_original_processed[:100], loaded_ids,
                                                       k=closest_k, metric='euclidean'))
print(index)
print(closest_indexes)

closest_indexes_dict = dict(zip(index, closest_indexes))
top_words_dict = dict(zip(loaded_ids, loaded_top_exemplar_words))
jaccard_distance_list = [[] for _ in range(len(loaded_ids))]
counter = 0

for i in loaded_ids:
    print(i)
    # print(top_exemplar_words[closest_indexes[i][0]])
    # print(top_exemplar_words[closest_indexes[i][closest_k-1]])
    tempList = list()
    instance = ' '.join(map(str, top_words_dict.get(i)))

    for j in range(closest_k):
        print(j)
        listToStr = ' '.join(map(str, top_words_dict.get(closest_indexes_dict[i][j])))
        print(listToStr)
        tempList.append(listToStr)

    for j in range(closest_k):
        jaccard_distance_list[counter].append(1 - get_jaccard_sim(instance, tempList[j]))

    counter += 1

instability_list = list()
for i in range(len(jaccard_distance_list)):
    print(i)
    instability_list.append(jaccard_distance_list[i][0]/jaccard_distance_list[i][closest_k-1])

print(instability_list)
print('Average instability: ', (sum(instability_list)/len(instability_list)))
print('Standard Deviation: ', stdev(instability_list))
