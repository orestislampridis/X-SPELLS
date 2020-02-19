import pickle
import string

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import load_model
import decision_tree
from lstm_vae import create_lstm_vae, inference
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial
from imblearn.over_sampling import SMOTE
from collections import Counter
import csv


def get_text_data(num_samples, data_path):
    thousandwords = [line.rstrip('\n') for line in open('data/1-1000.txt')]

    print('thousandwords', thousandwords)
    # vectorize the data
    input_texts = []
    input_texts_original = []
    input_words = set(["\t"])
    all_input_words = []
    lines = []

    df = pd.read_csv(data_path, encoding='utf-8')
    # Removing the offensive comments, keeping only neutral and hatespeech,
    # in order to convert the problem to a simple binary classification problem
    df = df[df['class'] != 1]
    X = df['tweet'].values
    y = df['class'].values
    class_names = ['hate', 'offensive', 'neutral']

    # filename = 'data/hate_stopwords_retained.csv'
    # X_new = []
    # with open(filename, 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         X_new.extend(row)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

    new_X_test = X_test

    for line in new_X_test:
        input_texts_original.append(line)
        lines.append(
            line.lower().translate(str.maketrans('', '', string.punctuation)))  # lowercase and remove punctuation
    print(lines)

    # with open(data_path, "r", encoding="utf-8") as f:
    # lines = f.read().lower().split("\n")
    # print(lines)

    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text = line

        input_text = word_tokenize(input_text)
        input_text.append("<end>")
        input_texts.append(input_text)

        for word in input_text:
            if word not in input_words:
                input_words.add(word)

        for word in input_text:  # This will be used to count the words and keep the most frequent ones
            all_input_words.append(word)

    words_to_keep = 4499
    most_common_words = [word for word, word_count in
                         Counter(all_input_words).most_common(words_to_keep)]  # Keep the 1000 most common words
    most_common_words.append('\t')

    for word in thousandwords:  # Here we add the 1000 most common english words
        most_common_words.append(word)

    print(most_common_words)

    input_texts_cleaned = [[word for word in text if word in most_common_words] for text in input_texts]

    # previous_final_input_words = sorted(list(input_words))
    final_input_words = sorted(list(set(most_common_words)))
    # total_num_encoder_tokens = len(previous_final_input_words)
    num_encoder_tokens = len(final_input_words)
    max_encoder_seq_length = max([len(txt) for txt in input_texts_cleaned]) + 1

    print("input_texts_cleaned", input_texts_cleaned)
    # print(previous_final_input_words)
    print(most_common_words)
    print(final_input_words)

    print("Number of samples:", len(input_texts_cleaned))
    # print("Number of total unique input tokens:", total_num_encoder_tokens)
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(final_input_words)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts_cleaned), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")
    decoder_input_data = np.zeros((len(input_texts_cleaned), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")

    for i, input_text_cleand in enumerate(input_texts_cleaned):
        decoder_input_data[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text_cleand):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
            decoder_input_data[i, t + 1, input_token_index[char]] = 1.0

    print('.......')
    for i in range(10):
        print(input_texts[i])
        print(input_texts_cleaned[i])
        print('')

    return max_encoder_seq_length, num_encoder_tokens, final_input_words, input_token_index, reverse_input_char_index, \
           encoder_input_data, decoder_input_data, input_texts_original


def decode(s):
    return inference.decode_sequence(s, gen, stepper, input_dim, char2id, id2char, max_encoder_seq_length)


def get_sentences():
    input_sentences = []
    state_input_sentences = []

    for i in range(len(encoder_input_data)):
        mean, variance = enc.predict([[encoder_input_data[i]]])
        seq = np.random.normal(size=(latent_dim,))
        seq = mean + variance * seq
        input_sentences.append(input_texts_original[i])
        state_input_sentences.append(seq)

    return input_sentences, state_input_sentences


def calculate_min_max(list):
    c = np.min(list, axis=0)
    d = np.max(list, axis=0)
    return c, d


def generate_sentences(number_of_sentences, number_of_max_attempts, number_of_random_sentences, probability):
    state_sentences = [[] for _ in range(number_of_sentences)]
    decoded_sentences = [[] for _ in range(number_of_sentences)]

    for i in range(number_of_sentences):

        print("sentence : ", i)
        seq_from = state_in_sentences[i]

        number_of_ticks = 0
        max_attempts = number_of_max_attempts
        random_sentences_to_create = number_of_random_sentences

        while (len(decoded_sentences[i]) < random_sentences_to_create) and number_of_ticks < max_attempts:

            newseq = np.copy(seq_from)
            for d in range(latent_dim):
                rm = np.random.random()
                if rm >= probability:
                    newseq[0, d] = (largest_x[0, d] - smallest_x[0, d]) * np.random.random() + smallest_x[0, d]

            if decode(newseq) not in decoded_sentences[i]:
                state_sentences[i].append(newseq)
                decoded_sentences[i].append(decode(newseq))

            print(len(decoded_sentences[i]))
            number_of_ticks += 1
            print(number_of_ticks)

    return state_sentences, decoded_sentences


def get_predictions(bb_filename, vect_filename, number_of_sentences):
    # Load the black box model from disk
    filename = bb_filename
    loaded_model = pickle.load(open(filename, 'rb'))

    # Load the TF-IDF vectorizer of the respective dataset
    vectorizer = pickle.load(open(vect_filename, 'rb'))

    test_vectors = [[] for _ in range(number_of_sentences)]
    preds = [[] for _ in range(number_of_sentences)]

    print('predictions are: ')

    final_unique_state_sentences = [[] for _ in range(number_of_sentences)]  # with initial sentence on first place
    final_unique_decoded_sentences = [[] for _ in range(number_of_sentences)]

    for i in range(number_of_sentences):
        # print(in_sentences[i])
        # print(state_in_sentences[i])
        # print(original_sent_preds[i])

        final_unique_state_sentences[i].append(state_in_sentences[i])
        final_unique_decoded_sentences[i].append(in_sentences[i])

        final_unique_state_sentences[i].extend(generated_state_sentences[i])
        final_unique_decoded_sentences[i].extend(generated_decoded_sentences[i])

        test_vectors[i] = vectorizer.transform(final_unique_decoded_sentences[i])
        # Predicting
        preds[i] = loaded_model.predict(test_vectors[i])

    return preds, final_unique_state_sentences, final_unique_decoded_sentences


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def calculate_weights(Z, metric):
    if np.max(Z) != 1 and np.min(Z) != 0:
        Zn = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        distances = cdist(Zn, Zn[0].reshape(1, -1), metric=metric).ravel()
    else:
        distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()
        weights = kernel(distances)
        return weights


if __name__ == "__main__":

    res = get_text_data(num_samples=20000, data_path="data/hate_tweets.csv")

    max_encoder_seq_length, num_enc_tokens, characters, char2id, id2char, \
    encoder_input_data, decoder_input_data, input_texts_original = res

    print(encoder_input_data.shape, "Creating model...")

    input_dim = encoder_input_data.shape[-1]
    timesteps = encoder_input_data.shape[-2]
    batch_size = 1
    latent_dim = 4878
    intermediate_dim = 353
    epochs = 200

    vae, enc, gen, stepper, vae_loss = create_lstm_vae(input_dim,
                                                       batch_size=batch_size,
                                                       intermediate_dim=intermediate_dim,
                                                       latent_dim=latent_dim)
    print("Training VAE model...")

    # vae.fit([encoder_input_data, decoder_input_data], encoder_input_data, epochs=epochs, verbose=1)
    # vae.save('models/hate_vae_model.h5', overwrite=True)
    # enc.save('models/hate_enc_model.h5', overwrite=True)
    # gen.save('models/hate_gen_model.h5', overwrite=True)
    # stepper.save('models/hate_stepper_model.h5', overwrite=True)

    del vae
    del enc
    del gen
    del stepper

    vae = load_model('models/hate_vae_model.h5', custom_objects={'vae_loss': vae_loss}, compile=False)
    enc = load_model('models/hate_enc_model.h5', compile=False)
    gen = load_model('models/hate_gen_model.h5', compile=False)
    stepper = load_model('models/hate_stepper_model.h5', compile=False)
    vae.summary()
    print("Fitted, predicting...")

    # For how many sentences we want to run X-SPELLS
    no_of_sentences = 5

    in_sentences, state_in_sentences = get_sentences()
    print(in_sentences)
    training_state_list = np.array(state_in_sentences)
    smallest_x, largest_x = calculate_min_max(training_state_list)

    generated_state_sentences, generated_decoded_sentences = generate_sentences(number_of_sentences=no_of_sentences,
                                                                                number_of_max_attempts=5000,
                                                                                number_of_random_sentences=200,
                                                                                probability=0.4)

    pickled_black_box_filename = 'models/hate_saved_rf_model_stopwords_retained.sav'
    pickled_vectorizer_filename = 'models/hate_tfidf_vectorizer.pickle'

    predictions, final_state_sentences, final_decoded_sentences = get_predictions(pickled_black_box_filename,
                                                                                  pickled_vectorizer_filename,
                                                                                  no_of_sentences)

    print(predictions)

    nbr_features = latent_dim
    idx = []
    fidelities = []
    bbpreds = []
    dtpreds = []
    exemplars = []
    topWords = []

    for i in range(no_of_sentences):

        y = []
        print(i)

        if len(final_decoded_sentences[i]) < 60:
            print(len(final_decoded_sentences[i]))
            print('Not enough random sentences.')
            continue

        oversampling = 0

        Z = final_state_sentences[i]
        Z_text = final_decoded_sentences[i]
        Yb = predictions[i]
        selectedText = list()

        Y0 = (np.count_nonzero(Yb == 0))
        Y2 = (np.count_nonzero(Yb == 2))

        if Y0 < 6 or Y2 < 6:
            print('Not enough samples for smote.')
            continue

        if Y0 / (Y0 + Y2) < 0.4 or Y2 / (Y0 + Y2) < 0.4:
            oversampling = True

        Z = np.array(Z)
        Yb = np.array(Yb)

        Z = Z.squeeze()  # convert from 3d to 2d

        YbReshaped = Yb.reshape(1, -1)
        ZReshaped = Z.reshape(1, -1)

        metric = 'cosine'  # 'euclidean'
        kernel_width = float(np.sqrt(nbr_features) * 0.75)
        kernel = default_kernel
        kernel = partial(kernel, kernel_width=kernel_width)

        weights = calculate_weights(ZReshaped, metric)
        class_values = ['0', '2']

        if oversampling is True:
            sm = SMOTE(random_state=42)
            print(np.unique(Yb, return_counts=True))
            Z, Yb = sm.fit_resample(Z, Yb)

        for t in range(len(Z)):
            y.append(np.expand_dims(Z[t], axis=0))  # convert from 2d to 3d

        for t in range(len(Z_text), len(Z)):
            Z_text.append(decode(y[t]))

        print(len(Z_text))
        print(len(Z))

        dt = decision_tree.learn_local_decision_tree(Z, Yb, weights, class_values, prune_tree=True)
        Yc = dt.predict(Z)
        print(Yc)

        leave_id = dt.apply(Z)
        print(leave_id)
        others_in_same_leaf = np.where(leave_id == leave_id[0])[0]
        print(others_in_same_leaf)

        print('original sentence: ', Z_text[0])
        print('all sentences: ', Z_text)
        print('black box prediction', Yb[0])
        print('decision tree prediction', Yc[0])

        fidelity = accuracy_score(Yb, Yc)

        print('fidelity', fidelity)  # fidelity
        nbr_exemplars = 5

        if (len(others_in_same_leaf) < nbr_exemplars):
            print('Not enough exemplars in the leaf')
            continue

        selected_exemplars = np.random.choice(others_in_same_leaf, size=nbr_exemplars, replace=False)

        print('exemplars:')
        for j in selected_exemplars:
            print(Z_text[j])
            selectedText.append(Z_text[j])

        selectedText = np.array(selectedText)

        number_of_words = 5
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(selectedText)

        vocabulary = vectorizer.get_feature_names()
        ind = np.argsort(X.toarray().sum(axis=0))[-number_of_words:]

        top_n_words = [vocabulary[a] for a in ind]

        print(top_n_words)

        idx.append(i)
        fidelities.append(fidelity)
        bbpreds.append(Yb[0])
        dtpreds.append(Yc[0])
        exemplars.append(selectedText)
        topWords.append(top_n_words)

        print('')

    for i in range(len(idx)):
        print(idx[i])
        print(final_decoded_sentences[i][0])
        print(fidelities[i])
        print(bbpreds[i])
        print(dtpreds[i])
        print(exemplars[i])
        print(topWords[i])

    with open('output/hate_RF.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(idx)):
            writer.writerow(
                [idx[i], final_decoded_sentences[i][0], bbpreds[i], dtpreds[i], fidelities[i],
                 exemplars[i], topWords[i]])
