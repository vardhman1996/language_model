'''
Vikram Sringari
CSE517
This file tests the model by reading in characters with the o command,
showing log probability of a character to show with q command, and generates
characters with the g command
'''
import sys, os
#reload(sys)
import io
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
#sys.setdefaultencoding("utf-8")
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import math as m
import numpy as np
from numpy.random import choice
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#gets log prob and generates characters (if generate is True)
def char_logprob(model, text, i, generate=False):

  # turns characters to integers

  chars = [i for i in range(65424)]
  char_en  =  []

  for char in text:

    char_en.append(ord(char))

  # truncate sequences to a 4 like the data sequences
  char_en = pad_sequences([char_en], maxlen=4, truncating='pre')
  
  # one hot encoded all the integers 
  char_en = to_categorical(char_en, num_classes=len(chars))
  
  char_en = char_en.reshape(1, char_en.shape[1],char_en.shape[2])
    
  # character prediction
  
  y_hat = model.predict_classes(char_en, verbose=0)
  # character probability distribution
  prob = model.predict_proba(char_en, verbose=0)
    
  out = ''
  output= ''
  
  #log probability     
  output = str(m.log(prob[0][ord(text[len(text)-1])],2 ))
  
  
  if generate:  

    draw = choice(chars, len(chars), p=prob[0])     
    output = chr(draw[0])
    
  return output
  
#parses input data  
def char_parser(model):
  text = input(":")
  text = text#.decode('utf8') #used for python 2
  cmds = list(text)
  hist = ''
  
  #goes through all commands like o, q, g, x
  for i in range(len(cmds)):
    if cmds[i] == 'o' and (cmds[i-1] != 'o' or (cmds[i-1] == 'o' and  cmds[i-2] == 'o')) :
      hist += cmds[i+1]
      if cmds[i+1] == u'\u0003':
        hist = '' #deletes history if stop character is presents
      print('// added a character to the history!')
    elif cmds[i] == 'q' and (cmds[i-1] != 'q' or (cmds[i-1] == 'q' and  cmds[i-2] == 'q')):  
      print(char_logprob(model, hist+cmds[i+1], i))
    elif cmds[i] == 'g' and cmds[i-1] != 'o' and cmds[i-1] != 'q':
      if len(hist) == 0:
        char = np.random.randint(0, high=65424)
        hist += chr(char)
        print(chr(char) + '// generated a character! prob of generation: ' +char_logprob(model, hist, i))
      else:	  
        char = char_logprob(model, hist, i, generate=True)
        hist += char
        print(char + '// generated a character! prob of generation: ' +char_logprob(model, hist, i))
    elif cmds[i] == 'x' and cmds[i-1] != 'o' and cmds[i-1] != 'q':
      break 
      
def main():
  # load the model
  model = load_model('model.h5')
  # load the mapping
  #mapping = load(open('mapping.pkl', 'rb'))
  char_parser(model)
  
if __name__ == "__main__":
  main()
  
  