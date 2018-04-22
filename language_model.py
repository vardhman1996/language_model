'''
Vikram Sringari
CSE517
This file makes the character based language model
This model is specifcally an LSTM
'''
from pickle import dump
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#reads parsed data file in
#takes filename parameter
def read_file(filename):
	file = open(filename, 'r')
	# read text
	text = file.read()
	file.close()
	return text

#computes model after adding layers compiling and fiting
#returns and saves model
def main():
  input_file = 'sequences.txt'
  text = read_file(input_file)
  text_lines = text.split('\n')
  
  # all character from data
  chars = [i for i in range(65424)]
  
  seqs = list()
  
  for line in text_lines:

  	chars_en = [ord(char) for char in line.decode('utf-8')]
  	# store
  	seqs.append(chars_en[0:5])
  
  #print(seqs)
  
  seqs[len(seqs)-1][4] = 3 #sets end character as last in all of sequences
  
  # vocab size
  V = len(chars)
  #print('Vocabulary Size: %d' % vocab_size)
  
  seqs = np.asarray(seqs)
  
  #sets inputs and outputs
  X, Y = seqs[:,:-1], seqs[:,-1]
  seqs = [to_categorical(x, num_classes=V) for x in X]
  X = np.asarray(seqs)
  Y = to_categorical(Y, num_classes=V)
  
  # set the model
  model = Sequential()
  model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2])))
  model.add(Dense(V, activation='softmax'))
  # compile the model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit the model
  model.fit(X, Y, epochs=150, verbose=2)
  
  model.save('model.h5')
  

if __name__ == "__main__":
  main()

