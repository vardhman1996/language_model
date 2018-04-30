import pandas as pd
import codecs
import numpy as np
import random
import csv


DATA = "data/sentences.csv"
DATA_TRAIN = "data/wili-2018/x_train.txt"
DATA_TEST = "data/wili-2018/x_test.txt"
MAX_SENTENCES = 100

def read_data_taboeta(datafile):
    sent_df = pd.read_csv(datafile, sep="\t", encoding="utf-8", names=['num', 'lang', 'sent'])
    trim_taboeta_sentences(sent_df)


def trim_taboeta_sentences(data_df):
    languages = data_df['lang'].unique()
    sentence_list = []
    for i, lang in enumerate(languages):

        print("Completed: {0} out of {1}".format(i, len(languages)))
        sentence_df = data_df.loc[data_df['lang'] == lang]
        data = sentence_df['sent'].values
        random.shuffle(data)
        sentence_list += data[:MAX_SENTENCES].tolist()
        if i != 0 and i % 1000 == 0:
            save_data(sentence_list)
            sentence_list = []

    save_data(sentence_list)


def read_data_wili(datafile):
    data = codecs.open(datafile, encoding='utf-8')
    sentence_list = []
    all_data = list(data)
    print("LEN OF FILE: ", len(all_data))
    random.shuffle(all_data)

    for i, sentence in enumerate(all_data):
        print("Completed: {0} out of 117500".format(i))
        sentence_list += [sentence.strip()]
        if i == 200:
            break
        if i != 0 and i % 500 == 0:
            save_data(sentence_list)
            sentence_list = []
    save_data(sentence_list)


def save_data(final_list):
    df = pd.DataFrame(np.array(final_list))
    with codecs.open('data/final_sentences.csv', 'a', 'utf-8') as file:
        df.to_csv(file, sep='\t', index=False, header=None)

def main():
    read_data_taboeta(DATA)
    read_data_wili(DATA_TRAIN)
    read_data_wili(DATA_TEST)


if __name__ == "__main__":
    main()