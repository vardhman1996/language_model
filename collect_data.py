import pandas as pd
import numpy as np
import random
import csv
import io


DATA = "data/sentences.csv"
DATA_TRAIN = "data/wili-2018/x_train.txt"
DATA_TEST = "data/wili-2018/x_test.txt"
MAX_LENGTH = 98 #[start + sentence + stop]

def read_data_taboeta(datafile):
    sent_df = pd.read_csv(datafile, sep="\t", encoding="utf-8", names=['num', 'lang', 'sent'])
    trim_taboeta_sentences(sent_df)


def trim_taboeta_sentences(data_df):
    languages = data_df['lang'].unique()
    sentence_list = []
    for i, lang in enumerate(languages):
        sentence_df = data_df.loc[data_df['lang'] == lang]
        data = sentence_df['sent'].values
        random.shuffle(data)

        for sentence in data:
            sentence_list += cut_off_sentence(sentence)
        if (i + 1) % 10 == 0:
            for s in sentence_list:
                if (len(s)) > MAX_LENGTH:
                    print("FUCK?")
            save_data(sentence_list)
            print("Completed: {0} out of {1}".format(i + 1, len(languages)))
            sentence_list = []

    if len(sentence_list) != 0:
        save_data(sentence_list)


def cut_off_sentence(sentence):
    if len(sentence) <= MAX_LENGTH:
        return [sentence]
    else:
        return [sentence[i:i + MAX_LENGTH] for i in range(0, len(sentence), MAX_LENGTH)]


def read_data_wili(datafile):
    data = io.open(datafile, 'r', encoding='utf-8')
    sentence_list = []
    all_data = list(data)
    print("LEN OF FILE: ", len(all_data))
    random.shuffle(all_data)

    for i, sentence in enumerate(all_data):
        sentence_list += cut_off_sentence(sentence.strip())
        if (i + 1) % 20000 == 0:
            save_data(sentence_list)
            print("Completed: {0} out of {1}".format(i + 1, len(all_data)))
            sentence_list = []

    if len(sentence_list) != 0:
        save_data(sentence_list)


def save_data(final_list):
    df = pd.DataFrame(np.array(final_list))
    with io.open('data/final_sentences.csv', 'a', encoding='utf-8') as file:
        df.to_csv(file, sep='\t', index=False, header=None)

def main():
    read_data_taboeta(DATA)
    read_data_wili(DATA_TRAIN)
    read_data_wili(DATA_TEST)


if __name__ == "__main__":
    main()