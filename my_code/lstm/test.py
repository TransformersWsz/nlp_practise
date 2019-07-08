#!/usr/bin/env python
# @Time    : 2019/7/8 19:42
# @Author  : Swift  
# @File    : test.py
# @Brief   : None
# @Link    : None


from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Dense, Flatten
from keras.preprocessing import sequence
import numpy as np

def lstm():
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    max_word = 400
    x_train = sequence.pad_sequences(x_train, maxlen=max_word)
    x_test = sequence.pad_sequences(x_test, maxlen=max_word)

    vocab_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_word))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=100, verbose=1)
    score = model.evaluate(x_test, y_test)
    print(score)


if __name__ == "__main__":
    lstm()