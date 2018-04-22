'''
Vikram Sringari
CSE517
This file parses the raw data file.
Then it produces sequences of length 5 for the LSTM
'''
import random
import io
#reads parsed data file in
#takes filename parameter
def read_file(ifile):
	file = io.open(ifile, 'r', encoding='utf-8')
	# reads text
	text = file.read()
	file.close()
	return text

# saves parsed file with all sequences 
def save_file(lines, ifile):
  text = '\n'.join(lines)
  file = io.open(ifile, 'w', encoding='utf-8')
  file.write(u'\ufeff')
  file.write(text)
  file.close()

#parses file to produce sequences of five 
def main():
  text = read_file('x_train.txt')
  
  # cleans data
  conects = text.split()
  text = ' '.join(conects)
  
  # turns in sequences of 5 charcters per line
  length = 5
  seqs = list()
  for i in range(length, len(text)):
  	# select sequence of tokens
  	seq = text[i-length:i+1]
  	# store
  	seqs.append(seq)
  #print('Total Sequences: %d' % len(seqs))
  #random.shuffle(seqs) #randomizes all data and selects portion
  seqs = seqs[(len(seqs)-1)/3000:10000+(len(seqs)-1)/3000]
  
  # saves to file
  output_file = 'sequences.txt'
  save_file(seqs, output_file)

if __name__ == "__main__":
  main()

