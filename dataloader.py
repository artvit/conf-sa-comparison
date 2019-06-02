import json
from random import shuffle

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from textpreprocessing import normalize_corpus

DEFAULT_TRAIN_FILENAME = 'train_data'
DEFAULT_TEST_FILENAME = 'test_data'
DEFAULT_MAX_WORDS = 30000


def resize_data(data, results, size=None):
    if size is None:
        return data, results
    size = int(size)
    datatype = data.dtype.name
    zipped = list(zip(list(data), list(results)))
    shuffle(zipped)
    resized = zipped[:size]
    result = list(zip(*resized))
    return np.array(result[0], dtype=datatype), np.array(result[1])


def save_to_file(filename, data, results):
    np.savez(filename, data, results)


def load_from_file(filename, size=None):
    content = np.load(filename + '.npz')
    data = content[content.files[0]]
    results = content[content.files[1]]
    print('Data is loaded')
    if size:
        return resize_data(np.array(data), np.array(results), size)
    dict = None
    with open(filename + '_dictionary.json') as dictionary_file:
        dict = json.load(dictionary_file)
    return data, results, dict


def create_file(train=True, max_num_words=DEFAULT_MAX_WORDS):
    filename = DEFAULT_TRAIN_FILENAME if train else DEFAULT_TEST_FILENAME
    csv_file = './training.csv' if train else './test.csv'
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df = pd.read_csv(csv_file, header=None, names=cols, encoding='latin-1')

    y = np.array(df['sentiment'], dtype=np.uint8)

    x = list(df['text'])
    x = normalize_corpus(x, text_lemmatization=False)
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    dictionary = tokenizer.word_index
    # Let's save this out so we can use it later
    with open(filename + '_dictionary.json', 'w') as dictionary_file:
        json.dump(dictionary, dictionary_file)

    save_to_file(filename, x, y)


def preprocess_results(y):
    classes = np.unique(y)
    class_dict = {}
    for i, c in enumerate(classes):
        class_dict[c] = i
    new_y = [class_dict[c] for c in y]
    return np.array(new_y, dtype=np.uint8)


def load_data(train=True, max_num_words=DEFAULT_MAX_WORDS):
    filename = DEFAULT_TRAIN_FILENAME if train else DEFAULT_TEST_FILENAME
    x, y, dict = load_from_file(filename)
    y = preprocess_results(y)
    y = to_categorical(y)
    return x, y


def test():
    # create_file()
    x, y = load_data()
    pass


if __name__ == '__main__':
    test()
    # main()
    # show_results()
