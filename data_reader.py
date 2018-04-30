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
            data_list += [[c for c in line]]
            if i == (self.batch_size - 1): break

        return data_list

    def format_data(self, char_list):
        batch_list = np.zeros((self.batch_size, len(char_list[0]), 32))
        for i, char_line in enumerate(char_list):
            data_element = np.zeros((len(char_line), 32))
            for j, c in enumerate(char_line):
                data_element[j] = self.char_to_bit(c)
            batch_list[i] = data_element
        return batch_list

    def char_to_bit(self, c):
        unpadded_bits = [int(i) for i in format(ord(c), 'b')]
        return [0] * (32 - len(unpadded_bits)) + unpadded_bits

    def get_data(self):
        batch_char_list = self.read_data()
        batch_char_bits = self.format_data(batch_char_list)

        return batch_char_bits