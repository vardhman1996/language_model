# import unittest
# from data_reader import DataReader
#
# FILENAME = 'data_file.csv'
# class DataReaderTest(unittest.TestCase):
#
#     def test_len_data(self):
#         batch_size = 10
#         dr = DataReader(FILENAME, batch_size=batch_size)
#         batch_list = dr.get_data()
#         print(batch_list.shape)
#         self.assertEqual(len(batch_list), 10)
#
#
#     def test_len_data_black_box(self):
#         batch_size = 1
#         dr = DataReader(FILENAME, batch_size=batch_size)
#         batch_list_1 = dr.get_data()
#         batch_list_2 = dr.get_data()
#         self.assertEqual(len(batch_list_1), len(batch_list_2))
