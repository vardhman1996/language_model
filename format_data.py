import io
import nltk
import pandas as pd
import numpy as np


STOP_CHAR = '\u0003' # U+0003 \x03
START_CHAR = '\u0002' # U+0002 \x02
FINAL_SENTENCES = 'data/final_sentences.csv'
def read_data(filename):
    final_sentences = io.open(filename, 'r', encoding='utf-8')
    all_sentence_chars = []
    for i, sentence in enumerate(final_sentences):
        sentence = sentence.strip()
        all_sentence_chars += [START_CHAR] + [c for c in sentence] + [STOP_CHAR]
        if i != 0 and i % 10000 == 0:
            print("Done ", i)
    return ''.join(all_sentence_chars)

def save_sentences(sentence_char_list):
    char_list = []
    for i, part in enumerate(sentence_char_list):
        char_list += [''.join(part)]
        if i != 0 and i % 10000 == 0:
            print("Saved: ", i)
            save_data(char_list)
    save_data(char_list)

def save_data(final_list):
    df = pd.DataFrame(np.array(final_list))
    with io.open('data/data_file.csv', 'a', encoding='utf-8') as file:
        df.to_csv(file, sep='\t', index=False, header=None)


def main():
    all_sentence_chars = read_data(FINAL_SENTENCES)
    char_list = list(nltk.ngrams(all_sentence_chars, 100))
    print("Length to save: ", len(char_list))
    save_sentences(char_list)

if __name__ == "__main__":
    main()