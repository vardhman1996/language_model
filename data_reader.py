import os
import io
import numpy as np

DATA_FOLDER = "data/"

class DataReader():
    def __init__(self, filename, batch_size=100):
        self.data_filename = os.path.join(DATA_FOLDER, filename)
        self.file = io.open(self.data_filename, 'r', encoding='utf-8')
        self.batch_size = batch_size

    def read_data(self):
        data_list = []
        for i, line in enumerate(self.file):
            line = line.strip()
            data = [c for c in line]
            if len(data) != 100: continue
            data_list += [data]
            if (i + 1) % self.batch_size == 0:
                yield data_list
                data_list = []

    def format_data(self, char_list):
        batch_list = np.zeros((self.batch_size, len(char_list[0])))
        # print(batch_list.shape)
        for i, char_line in enumerate(char_list):
            data_element = np.zeros((len(char_line)))
            for j, c in enumerate(char_line):
                data_element[j] = self.char_to_bit(c)
            batch_list[i] = data_element
        return batch_list.astype('int32')

    def char_to_bit(self, c):
        # unpadded_bits = [int(i) for i in format(ord(c), 'b')]

        return ord(c)

    def get_data(self):
        for batch_char_list in self.read_data():
            batch_char_bits = self.format_data(batch_char_list)
            yield batch_char_bits[:, :-1], batch_char_bits[:, -1]
