# coding: utf-8

import numpy as np

np.random.seed(13)

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from datetime import datetime

import gensim
import os

# path = './hw05_dataset/Document_hw5/VOM19980220.0700.0166'
# corpus = open(path).readlines()[3:200]
# corpus = [sentence.replace('\n', '').replace('-1', '').strip().split() for sentence in corpus if sentence.count(' ') >= 2]

file_names = []
folder_path = '../dataset/Document_hw5'
files = os.listdir(folder_path)

for f in files:
    fullpath = os.path.join(folder_path, f)
    if os.path.isfile(fullpath):
        file_names.append(fullpath)

# 減少文章
d_p = np.random.choice(2265, 2265, replace=False)
file_names = [f for i, f in enumerate(file_names) if i in d_p]

docs = []
for fn in file_names:
    temp = open(fn).readlines()[3:200]
    temp = [sentence.replace('\n', '').replace('-1', '').strip().split() for sentence in temp if
            sentence.count(' ') >= 2]
    docs.append(temp)

corpus = []
for doc in docs:
    for sen in doc:
        corpus.append(sen)
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(corpus)
# corpus = tokenizer.texts_to_sequences(corpus)
nb_samples = sum(len(s) for s in corpus)
V = 51253
dim = 100
window_size = 3


# corpus = np.array(corpus)

def generate_data(corpus, window_size, V):
    maxlen = window_size * 2
    for i, words in enumerate(corpus):
        contexts = []
        labels = []
        L = len(words)
        for index, word in enumerate(words):
            s = index - window_size
            e = index + window_size + 1

            contexts.append([words[i] for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(int(word))

            x = sequence.pad_sequences(contexts, maxlen=maxlen)
            y = np_utils.to_categorical(labels, V)

            yield (x, y)


cbow = Sequential()
cbow.add(Embedding(input_dim=V, output_dim=dim, input_length=window_size * 2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
cbow.add(Dense(V, activation='softmax'))

cbow.compile(loss='categorical_crossentropy', optimizer='adadelta')


def outputEmbeddingWeight(cbow, i):
    fn = 'embeding_' + str(datetime.now()) + '_' + str(i)
    fn = fn.replace("-", "").replace(" ", "").replace(":", "").replace(".", "")
    f = open(fn, 'w')
    # f.write(' '.join([str(V-1), str(dim)]))
    # f.write('\n')

    vectors = cbow.get_weights()[0]
    for i in range(V):
        f.write(','.join(map(str, list(vectors[i, :]))))
        f.write('\n')
    f.close()


for ite in range(9999999):
    loss = 0.
    for x, y in generate_data(corpus, window_size, V):
        loss += cbow.train_on_batch(x, y)
    outputEmbeddingWeight(cbow, ite)
    print(ite, loss)
