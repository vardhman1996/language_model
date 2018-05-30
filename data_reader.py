import os
import io
import numpy as np
import pickle as pkl
import tensorflow as tf

DATA_FOLDER = "data/"
NUM_SENTENCES = 7560180 # total number of sentences collected
# [START_CHAR + sentence + STOP_CHAR]
MAX_TIME_STEPS = 38
STOP_CHAR = '\u0003' # U+0003 \x03
START_CHAR = '\u0002' # U+0002 \x02
CHAR_TO_NUM_DICT = 'char_to_num.pkl'
NUM_TO_CHAR_DICT = 'num_to_char.pkl'

MAX_LENGTH = 25
RANDOM_WINDOW = 15

prob_window = [0.0708]*14
prob_window.append(1-0.0708*14)


def char_to_bit(c):
    unpadded_bits = [int(i) for i in format(ord(c), 'b')]
    return [0] * (32 - len(unpadded_bits)) + unpadded_bits

class DataReader:
    def __init__(self, filename, batch_size=128):
        self.data_filename = os.path.join(DATA_FOLDER, filename)
        self.file = io.open(self.data_filename, 'r', encoding='utf-8')
        self.batch_size = batch_size
        self.stop_char_bits = char_to_bit(STOP_CHAR)
        self.char_to_num = pkl.load(open(os.path.join(DATA_FOLDER, CHAR_TO_NUM_DICT), 'rb'))
        self.num_to_char = pkl.load(open(os.path.join(DATA_FOLDER, NUM_TO_CHAR_DICT), 'rb'))

    def get_char_to_num(self, char):
        if char not in self.char_to_num:
            return len(self.char_to_num)

        return self.char_to_num[char]

    def get_num_to_char(self, num):
        return self.num_to_char[num]

    def read_data(self):
        data_list = []
        num_point = 0
        for line in self.file:
            line = line.strip()
            # some weird issue where the line length is more than MAX_TIME_STEPS.
            # need to remove all the extra quotes that appear when saving the file.
            line = line.strip('\"')
            line = line.replace('""', '"')

            data = [c for c in line]
            if len(data) < MAX_TIME_STEPS:
                continue
            data_list += [data]
            num_point += 1
            if num_point % self.batch_size == 0:
                return data_list

    def format_data(self, char_list):
        batch_list_x = np.zeros((self.batch_size, MAX_LENGTH, 32))
        batch_list_y = np.zeros((self.batch_size,))
        for i, char_line in enumerate(char_list):
            char_line = [START_CHAR] + char_line + [STOP_CHAR]
            batch_instance_x = np.zeros((MAX_LENGTH, 32))

            #slice_index = np.random.randint(0, RANDOM_WINDOW)
            slice_index = np.random.choice(RANDOM_WINDOW, p = prob_window) 
            sliced_char_line = char_line[slice_index:slice_index + MAX_LENGTH]

            batch_instance_y = self.char_to_num[char_line[slice_index + MAX_LENGTH]]

            for j, c in enumerate(sliced_char_line):
                batch_instance_x[j] = char_to_bit(c)

            batch_instance_x[len(sliced_char_line):] = self.stop_char_bits

            assert(batch_instance_y != None)
            batch_list_x[i] = batch_instance_x
            batch_list_y[i] = batch_instance_y

        return batch_list_x, batch_list_y

    def get_data(self, num_batches=60000):
        for batch in range(num_batches):
            batch_char_list = self.read_data()
            if batch_char_list == None:
                self.file.seek(0)
                batch_char_list = self.read_data()

            batch_x, batch_y = self.format_data(batch_char_list)
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=(len(self.char_to_num) + 1))

            yield batch_x, batch_y


# EXAMPLE USAGE
# dr = DataReader('final_sentences.csv', batch_size=1024)
# for i, (batch_x, batch_y)  in enumerate(dr.get_data(num_batches=11031)):
#     print(i, batch_x.shape, batch_y.shape)
