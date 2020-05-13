"""
Train a RF black box model for the hate speech dataset.

Also calculate fidelity of LIME explanations when using the RF used for the fidelity experiment
"""

import csv
import pickle
import re
import string

import numpy as np
import pandas as pd
import sklearn
from sklearn import feature_extraction
from lime.lime_text import LimeTextExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def cleanText(var):
    # replace punctuation with spaces
    var = re.sub('[{}]'.format(string.punctuation), " ", var)
    # remove double spaces
    var = re.sub(r'\s+', " ", var)
    # put in lower case
    var = var.lower().split()
    # remove words that are smaller than 3 characters
    var = [w for w in var if len(w) >= 3]
    # remove stop-words
    #var = [w for w in var if w not in stopwords.words('english')]
    # stemming
    #stemmer = nltk.PorterStemmer()
    #var = [stemmer.stem(w) for w in var]
    var = " ".join(var)
    return var


# Removes 'rt' from all input data
def my_clean(text):
    text = text.lower().split()
    text = [w for w in text]
    text = " ".join(text)
    text = re.sub(r"rt", "", text)
    return text


def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text


def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def preProcessing(strings):
    clean_tweet_texts = []
    for string in strings:
        clean_tweet_texts.append(my_clean(strip_all_entities(strip_links(string))))
        #clean_tweet_texts.append(my_clean(string))
    return clean_tweet_texts


def calculate_fidelity():
    # Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on
    # vectorized representation of texts (tf-idf in this case). For this purpose, we will use
    # sklearn's pipeline, and thus implement predict_proba on raw_text lists.
    c = make_pipeline(vectorizer, loaded_model)
    print(c.predict_proba)

    # Creating an explainer object. We pass the class_names as an argument for prettier display.
    explainer = LimeTextExplainer(class_names=class_names)

    ids = list()
    fidelities = list()

    for i in range(len(X_test)):
        print('index', i)
        # Generate an explanation with at most n features for a random document in the test set.
        idx = i
        exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)

        label = loaded_model.predict(test_vectors[idx])[0]
        label = label // 2
        print(label)
        bb_probs = explainer.Zl[:, label]
        print('bb_probs: ', bb_probs)
        lr_probs = explainer.lr.predict(explainer.Zlr)
        print('lr_probs: ', lr_probs)
        fidelity = 1 - np.sum(np.abs(bb_probs - lr_probs) < 0.01) / len(bb_probs)
        print('fidelity: ', fidelity)
        ids.append(i)
        fidelities.append(fidelity)

    fidelity_average = 0

    for i in range(len(ids)):
        print(ids[i])
        print(fidelities[i])
        fidelity_average += fidelities[i]

    print("fidelity average is: ", fidelity_average / len(ids))

    with open('output/LIME_hs_RF.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ids)):
            writer.writerow([ids[i], 'hate speech', 'RF', fidelities[i]])


df = pd.read_csv("data/hate_tweets.csv", encoding='utf-8')
# Removing the offensive comments, keeping only neutral and hatespeech,
# in order to convert the problem to a simple binary classification problem
df = df[df['class'] != 1]
X = df['tweet'].values
y = df['class'].values
class_names = ['hate', 'offensive', 'neutral']

X = preProcessing(X)

filename = 'data/hate_stopwords_retained.csv'

# Use the following to avoid reloading the data every time
#with open(filename, 'w') as resultFile:
#    wr = csv.writer(resultFile, dialect='excel')
#    wr.writerow(X)

#X_new = []
#with open(filename, 'r') as f:
#    reader = csv.reader(f)
#    for row in reader:
#       X_new.extend(row)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

# We'll use the TF-IDF vectorizer, commonly used for text.
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf='false')
train_vectors = vectorizer.fit_transform(X_train)
pickle.dump(vectorizer, open("models/hate_tfidf_vectorizer.pickle", "wb"))
test_vectors = vectorizer.transform(X_test)

# Using random forest for classification.
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
         max_depth=1000, max_features=1000, max_leaf_nodes=None,
         min_impurity_decrease=0.0, min_impurity_split=None,
         min_samples_leaf=4, min_samples_split=10,
         min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
         oob_score=False, random_state=None, verbose=0,
         warm_start=False)

'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
'''

rf.fit(train_vectors, y_train)

# save the model to disk
filename = 'models/hate_saved_RF_model.sav'
pickle.dump(rf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
pred = loaded_model.predict(test_vectors)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
calculate_fidelity()
