import pandas as pd
import numpy as np
import random
import csv
import io
import os
import pickle as pkl

DATA = "data/sentences.csv"
DATA_TRAIN = "data/wili-2018/x_train.txt"
DATA_TEST = "data/wili-2018/x_test.txt"
DATA_PATH = 'data'
STOP_CHAR = '\u0003' # U+0003 \x03
UNK_CHAR = '\u0001' # UNK char


def read_data_taboeta(datafile):
    sent_df = pd.read_csv(datafile, sep="\t", encoding="utf-8", names=['num', 'lang', 'sent'])
    return trim_taboeta_sentences(sent_df)


def trim_taboeta_sentences(data_df):
    languages = data_df['lang'].unique()
    char_set = set()
    sum = 0.0
    sent = 0.0

    num_less_than = 0

    for i, lang in enumerate(languages):
        sentence_df = data_df.loc[data_df['lang'] == lang]
        data = sentence_df['sent'].values
        random.shuffle(data)

        for line in data:
            sum += len(line)
            sent += 1

            if len(line) < 30:
                num_less_than += 1

            char_set.update(c for c in line)

        if (i + 1) % 10 == 0:
            print("Completed: {0} out of {1}".format(i + 1, len(languages)))
    print("Num unique chars in Taboeta: {}".format(len(char_set)))

    print("AVG = ", sum / sent)
    print("NUM less = ", num_less_than)
    return char_set

def read_data_wili(datafile):
    data = io.open(datafile, 'r', encoding='utf-8')
    char_set = set()
    all_data = list(data)
    print("LEN OF FILE: ", len(all_data))
    random.shuffle(all_data)

    for i, sentence in enumerate(all_data):
        char_set.update(c for c in sentence.strip())
        if (i + 1) % 20000 == 0:
            print("Completed: {0} out of {1}".format(i + 1, len(all_data)))
    print("Num unique chars in wili: {}".format(len(char_set)))
    return char_set

def make_mapping(char_set):
    char_to_num_map = dict()
    num_to_char_map = dict()
    for i, c in enumerate(char_set):
        num_to_char_map[i] = c
        char_to_num_map[c] = i

    return char_to_num_map, num_to_char_map


def save_map(map, filepath):
    with open(filepath, 'wb') as file:
        pkl.dump(map, file)


def main():
    universal_char_set = set()
    universal_char_set = universal_char_set | read_data_taboeta(DATA)
    universal_char_set = universal_char_set | read_data_wili(DATA_TRAIN)
    universal_char_set = universal_char_set | read_data_wili(DATA_TEST)

    universal_char_set.add(STOP_CHAR)
    universal_char_set.add(UNK_CHAR)

    print("Num unique chars: {}".format(len(universal_char_set)))

    char_to_num_map, num_to_char_map = make_mapping(universal_char_set)
    save_map(char_to_num_map, os.path.join(DATA_PATH, 'char_to_num.pkl'))
    save_map(num_to_char_map, os.path.join(DATA_PATH, 'num_to_char.pkl'))


if __name__ == '__main__':
    main()