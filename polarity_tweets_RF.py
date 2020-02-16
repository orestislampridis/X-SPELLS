import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
from lime.lime_text import LimeTextExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import RandomizedSearchCV
import csv


def cleanText(var):
    # replace punctuation with spaces
    var = re.sub('[{}]'.format(string.punctuation), " ", var)
    # remove double spaces
    var = re.sub(r'\s+', " ", var)
    # put in lower case
    var = var.lower().split()
    # remove words that are smaller than 2 characters
    var = [w for w in var if len(w) >= 3]
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


df = pd.read_csv("data/polarity_tweets.csv", encoding='utf-8')
#X = df['tweet'].values
y = df['class'].values
class_names = ['negative', 'positive']

# X = preProcessing(X)

filename = 'data/polarity_stopwords_retained.csv'
# with open(filename, 'w') as resultFile:
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
#pickle.dump(vectorizer, open("models/polarity_tfidf_vectorizer.pickle", "wb"))
test_vectors = vectorizer.transform(X_test)

# Using random forest for classification.
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                            max_depth=1000, max_features=1000, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=4, min_samples_split=10,
#                            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
#                            oob_score=False, random_state=None, verbose=0,
#                            warm_start=False)

#rf.fit(train_vectors, y_train)

# Save the model to disk
filename = 'models/polarity_saved_rf_model_stopwords_retained.sav'
# pickle.dump(rf, open(filename, 'wb'))

# Load the model from disk
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

for i, e in enumerate(X_test):
    print(str(i + 1) + '.', e)

for i in range(len(X_test)):
    #print(i)
    # Generate an explanation with at most n features for a random document in the test set.
    idx = i
    exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)
    # print('Document id: %d' % idx)
    # print('Probability(neutral) =', c.predict_proba([X_test[idx]])[0, 1])
    # print('True class: %s' % class_names[y_test[idx]])

    label = loaded_model.predict(test_vectors[idx])[0]
    #print(label)
    bb_probs = explainer.Zl[:, label]
    lr_probs = explainer.lr.predict(explainer.Zlr)
    fidelity = 1 - np.sum(np.abs(bb_probs - lr_probs) < 0.01) / len(bb_probs)
    ids.append(i)
    fidelities.append(fidelity)

fidelityMO = 0

for i in range(len(ids)):
    # print(ids[i])
    # print(fidelities[i])
    fidelityMO += fidelities[i]

print("fidelityMO is: ", fidelityMO / len(ids))

with open('LIME.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(ids)):
        writer.writerow([ids[i], 'hate speech', 'RF', fidelities[i]])

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