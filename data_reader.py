import os
import io
import numpy as np

DATA_FOLDER = "data/"
NUM_SENTENCES = 7560180 # total number of sentences collected
# [START_CHAR + sentence + STOP_CHAR]
MAX_TIME_STEPS = 98
STOP_CHAR = '\u0003' # U+0003 \x03
START_CHAR = '\u0002' # U+0002 \x02

class DataReader:
    def __init__(self, filename, batch_size=128):
        self.data_filename = os.path.join(DATA_FOLDER, filename)
        self.file = io.open(self.data_filename, 'r', encoding='utf-8')
        self.batch_size = batch_size
        self.stop_char_bits = self.char_to_bit(STOP_CHAR)

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
            data_list += [data]
            num_point += 1
            if num_point % self.batch_size == 0:
                return data_list

    def format_data(self, char_list):
        batch_list = np.zeros((self.batch_size, MAX_TIME_STEPS + 2, 32))
        # print(batch_list.shape)
        for i, char_line in enumerate(char_list):
            char_line = [START_CHAR] + char_line + [STOP_CHAR]
            batch_instance = np.zeros((MAX_TIME_STEPS + 2, 32))
            for j, c in enumerate(char_line):
                batch_instance[j] = self.char_to_bit(c)

            batch_instance[len(char_line):] = self.stop_char_bits
            batch_list[i] = batch_instance
        return batch_list


    def char_to_bit(self, c):
        unpadded_bits = [int(i) for i in format(ord(c), 'b')]
        return [0] * (32 - len(unpadded_bits)) + unpadded_bits

    def get_data(self, num_batches=60000):
        for batch in range(num_batches):
            batch_char_list = self.read_data()
            if batch_char_list == None:
                self.file.seek(0)
                batch_char_list = self.read_data()

            batch_char_bits = self.format_data(batch_char_list)
            yield batch_char_bits


# EXAMPLE USAGE
# dr = DataReader('final_sentences.csv', batch_size=128)
# for i, batch in enumerate(dr.get_data(num_batches=100000)):
#     print(i, batch.shape)