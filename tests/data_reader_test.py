import unittest
from data_reader.data_reader import DataReader

FILENAME = 'final_sentences.csv'
class DataReaderTest(unittest.TestCase):

    def test_len_data(self):
        batch_size = 10
        dr = DataReader(FILENAME, batch_size=batch_size)
        batch_list = dr.get_data()
        self.assertEqual(len(batch_list), 10)


    def test_file_iterator(self):
        batch_size = 1
        dr = DataReader(FILENAME, batch_size=batch_size)
        batch_list_1 = dr.get_data()
        batch_list_2 = dr.get_data()
        self.assertNotEqual(batch_list_1, batch_list_2)
