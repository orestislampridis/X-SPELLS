"""
Train a DNN black box model for the polarity dataset.

Also calculate fidelity of LIME explanations when using the DNN used for the fidelity experiment
"""

import csv
import pickle
import re
import string

import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from DNN_base import TextsToSequences, Padder, create_model


def cleanText(var):
    # replace punctuation with spaces
    var = re.sub('[{}]'.format(string.punctuation), " ", var)
    # remove double spaces
    var = re.sub(r'\s+', " ", var)
    # put in lower case
    var = var.lower().split()
    # remove words that are smaller than 2 characters
    var = [w for w in var if len(w) >= 2]
    # remove stop-words
    # var = [w for w in var if w not in stopwords.words('english')]
    # stemming
    # stemmer = nltk.PorterStemmer()
    # var = [stemmer.stem(w) for w in var]
    var = " ".join(var)
    return var


def preProcessing(pX):
    clean_tweet_texts = []
    for t in pX:
        clean_tweet_texts.append(cleanText(t))
    return clean_tweet_texts


def calculate_fidelity():
    # Creating an explainer object. We pass the class_names as an argument for prettier display.
    explainer = LimeTextExplainer(class_names=class_names)

    ids = list()
    fidelities = list()

    for i, e in enumerate(X_test):
        print(str(i + 1) + '.', e)

    for i in range(len(X_test)):
        print('index: ', i)
        # Generate an explanation with at most n features for a random document in the test set.
        idx = i
        exp = explainer.explain_instance(X_test[idx], loaded_model.predict_proba, num_features=10)
        label = pred[i]
        label = label // 2

        bb_probs = explainer.Zl[:, label]
        print('bb_probs: ', bb_probs)
        lr_probs = explainer.lr.predict(explainer.Zlr)
        print('lr_probs: ', lr_probs)
        fidelity = 1 - np.sum(np.abs(bb_probs - lr_probs) < 0.01) / len(bb_probs)
        print('fidelity: ', fidelity)
        print('np.sum: ', np.sum(np.abs(bb_probs - lr_probs) < 0.01))
        ids.append(i)
        fidelities.append(fidelity)
        print('')

    fidelity_average = 0

    for i in range(len(ids)):
        print(ids[i])
        print(fidelities[i])
        fidelity_average += fidelities[i]

    print("fidelity average is: ", fidelity_average / len(ids))

    with open('output/LIME_po_DNN.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ids)):
            writer.writerow([ids[i], 'polarity', 'DNN', fidelities[i]])


df = pd.read_csv("polarity_tweets.csv", encoding='utf-8')
# Removing the offensive comments, keeping only neutral and hatespeech,
# in order to convert the problem to a simple binary classification problem
X = df['tweet'].values
y = df['class'].values
class_names = ['negative', 'positive']

X = preProcessing(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

sequencer = TextsToSequences(num_words=35000)
padder = Padder(140)
myModel = KerasClassifier(build_fn=create_model, epochs=100)

pipeline = make_pipeline(sequencer, padder, myModel)
pipeline.fit(X_train, y_train)

# Save the model to disk
filename = 'models/polarity_saved_DNN_model.sav'
pickle.dump(pipeline, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
# pred = pipeline.predict(X_test)
pred = loaded_model.predict(X_test)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
calculate_fidelity()
