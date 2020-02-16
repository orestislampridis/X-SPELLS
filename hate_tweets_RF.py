import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
from lime.lime_text import LimeTextExplainer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm, neighbors, tree, naive_bayes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np
from collections import OrderedDict
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV
import csv


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


def preProcessing(pX):
    clean_tweet_texts = []
    for t in pX:
        clean_tweet_texts.append(cleanText(t))
    return clean_tweet_texts


df = pd.read_csv("data/hate_tweets.csv", encoding='utf-8')
# Removing the offensive comments, keeping only neutral and hatespeech,
# in order to convert the problem to a simple binary classification problem
df = df[df['class'] != 1]
#X = df['tweet'].values
y = df['class'].values
class_names = ['hate', 'offensive', 'neutral']

#X = preProcessing(X)

filename = 'data/hate_stopwords_retained.csv'
#with open(filename, 'w') as resultFile:
#    wr = csv.writer(resultFile, dialect='excel')
#    wr.writerow(X)

X_new = []
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
       X_new.extend(row)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=42, stratify=y, test_size=0.25)

# We'll use the TF-IDF vectorizer, commonly used for text.
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf='false')
train_vectors = vectorizer.fit_transform(X_train)
#pickle.dump(vectorizer, open("models/hate_tfidf_vectorizer.pickle", "wb"))
test_vectors = vectorizer.transform(X_test)

# Using random forest for classification.
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#         max_depth=1000, max_features=1000, max_leaf_nodes=None,
#         min_impurity_decrease=0.0, min_impurity_split=None,
#         min_samples_leaf=4, min_samples_split=10,
#         min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
#         oob_score=False, random_state=None, verbose=0,
#         warm_start=False)

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

#rf.fit(train_vectors, y_train)

# save the model to disk
filename = 'models/hate_saved_rf_model_stopwords_retained.sav'
#pickle.dump(rf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
pred = loaded_model.predict(test_vectors)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
"""
# Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on
# vectorized representation of texts (tf-idf in this case). For this purpose, we will use
# sklearn's pipeline, and thus implement predict_proba on raw_text lists.
c = make_pipeline(vectorizer, loaded_model)

# Creating an explainer object. We pass the class_names as an argument for prettier display.
explainer = LimeTextExplainer(class_names=class_names)

ids = list()
fidelities = list()

for i in range(len(X_test)):

    print(i)
    # Generate an explanation with at most n features for a random document in the test set.
    idx = i
    exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)
    #print('Document id: %d' % idx)
    #print('Probability(neutral) =', c.predict_proba([X_test[idx]])[0, 1])
    #print('True class: %s' % class_names[y_test[idx]])

    label = loaded_model.predict(test_vectors[idx])[0]
    label = label//2
    print(label)
    bb_probs = explainer.Zl[:, label]
    print('bb_probs: ', bb_probs)
    print(len(bb_probs))
    lr_probs = explainer.lr.predict(explainer.Zlr)
    print('lr_probs: ', lr_probs)
    print(len(lr_probs))
    fidelity = 1 - np.sum(np.abs(bb_probs - lr_probs) < 0.01) / len(bb_probs)
    print('fidelity: ', fidelity)
    ids.append(i)
    fidelities.append(fidelity)

fidelityMO = 0

for i in range(len(ids)):
    print(ids[i])
    print(fidelities[i])
    fidelityMO += fidelities[i]

print("fidelityMO is: ", fidelityMO/len(ids))

with open('LIME.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(ids)):
        writer.writerow([ids[i], 'hate speech', 'RF', fidelities[i]])

for i, e in enumerate(X_test):
    print(str(i + 1) + '.', e)


# The explanation is presented below as a list of weighted features along with a graph and the initial document.
print(exp.as_list())
weights = OrderedDict(exp.as_list())
lime_w = pd.DataFrame({'words': list(weights.keys()), 'weights': list(weights.values())})
sns.barplot(x="words", y="weights", data=lime_w)
print(lime_w)
plt.xticks(rotation=45)
plt.title('Instance No{} features weights given by Lime'.format(idx))
print(X_test[idx])
plt.show()
"""