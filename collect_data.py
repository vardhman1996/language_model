import pandas as pd
import numpy as np
import random
import csv
import io


DATA = "data/sentences.csv"
DATA_TRAIN = "data/wili-2018/x_train.txt"
DATA_TEST = "data/wili-2018/x_test.txt"
MAX_LENGTH = 38 #[start + sentence + stop]
OUTPUT = 'data/final_sentences.csv'
SENTENCE_LENGTH_THRESH = 10



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
            if len(sentence) < SENTENCE_LENGTH_THRESH:
                continue
            sentence_list += cut_off_sentence(sentence)
        if (i + 1) % 1 == 0:
            for s in sentence_list:
                if (len(s)) < MAX_LENGTH:
                    print("FUCK?")
            save_data(sentence_list, 'a')
            print("Completed: {0} out of {1}".format(i + 1, len(languages)))
            sentence_list = []

    if len(sentence_list) != 0:
        save_data(sentence_list, 'a')


def cut_off_sentence(sentence):
    if len(sentence) <= MAX_LENGTH:
        new_sentence = sentence
        while len(new_sentence) < MAX_LENGTH:
            new_sentence += ' '
            for i in range(len(sentence)):
                if len(new_sentence) >= MAX_LENGTH: break
                new_sentence += sentence[i]
        return [new_sentence]
    else:
        sentence_chunks = [sentence[i:i + MAX_LENGTH] for i in range(0, len(sentence), MAX_LENGTH)]
        ret_sentences = []
        for sent in sentence_chunks:
            if len(sent) < SENTENCE_LENGTH_THRESH:
                continue
            ret_sentences += cut_off_sentence(sent)
        return ret_sentences


def read_data_wili(datafile):
    data = io.open(datafile, 'r', encoding='utf-8')
    sentence_list = []
    all_data = list(data)
    print("LEN OF FILE: ", len(all_data))
    random.shuffle(all_data)

    for i, sentence in enumerate(all_data):
        sentence = sentence.strip()
        if len(sentence) < SENTENCE_LENGTH_THRESH:
            continue
        sentence_list += cut_off_sentence(sentence)
        if (i + 1) % 20000 == 0:
            save_data(sentence_list, 'a')
            print("Completed: {0} out of {1}".format(i + 1, len(all_data)))
            sentence_list = []

    if len(sentence_list) != 0:
        save_data(sentence_list, 'a')


def randomize_data(datafilepath):
    sent_df = pd.read_csv(datafilepath, sep='\t', encoding="utf-8", names=['sent'])
    sentences = sent_df['sent'].values

    arr = np.zeros(40)
    temp_sentences = []
    for sent in sentences:
        if len(sent) < 38:
            continue
        temp_sentences += [sent]
        arr[len(sent) - 1] += 1

    print(arr)
    np.random.shuffle(np.array(temp_sentences))
    save_data(np.asarray(temp_sentences), 'w')

def save_data(final_list, mode):
    df = pd.DataFrame(np.array(final_list))
    with io.open(OUTPUT, mode, encoding='utf-8') as file:
        df.to_csv(file, sep='\t', index=False, header=None)

def main():
    # read_data_taboeta(DATA)
    read_data_wili(DATA_TRAIN)
    read_data_wili(DATA_TEST)
    randomize_data(OUTPUT)


if __name__ == "__main__":
    main()