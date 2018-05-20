'''
Vikram Sringari
CSE517
This file makes the character based language model
This model is specifcally an LSTM
'''
import os
from pickle import dump
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
import data_reader




# reads parsed data file in
# takes filename parameter
def read_file(filename):
    file = open(filename, 'r')
    # read text
    text = file.read()
    file.close()
    return text

#computes model after adding layers compiling and fiting
#returns and saves model
def main():
    V = 136755
    model = Sequential()
    dr = data_reader.DataReader('data_file.csv', batch_size=10)
    for (batch_x, batch_y) in dr.get_data():
        X = batch_x
        Y = batch_y
        seqs = [to_categorical(x, num_classes=V) for x in X]
        X = np.asarray(seqs)
        Y = to_categorical(Y, num_classes=V)
        # set the model

        # print(X.shape, Y.shape)
        model.add(LSTM(1, input_shape=(X.shape[1], X.shape[2])))
        model.add(Flatten())
        model.add(Dense(V, activation='softmax'))
        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the model
        model.fit(X, Y, epochs=1, verbose=2)

    model.save('model.h5')
  

if __name__ == "__main__":
  main()

