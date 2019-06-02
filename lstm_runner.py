from lstm import RNN
import os, pickle, re, string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers import Embedding
from keras import optimizers
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras_tqdm import TQDMCallback


MAX_NUM_WORDS   = 25000
EMBEDDING_DIM   = 300
MAX_SEQ_LENGTH  = 500
USE_GLOVE       = False
LSTM_SIZE       = 200
LSTM_LAYERS     = 1
DROPOUT_RATE    = 0.1
HIDDEN_UNITS    = 200
NB_CLASSES      = 2

# LEARNING
BATCH_SIZE      = 100
NB_EPOCHS       = 6
RUNS            = 3
VAL_SIZE        = 0.3


def clean_doc(doc):
    """
    Cleaning a document by several methods:
        - Lowercase
        - Removing whitespaces
        - Removing numbers
        - Removing stopwords
        - Removing punctuations
        - Removing short words
    """
    ###stop_words = set(stopwords.words('english'))

    # Lowercase
    doc = doc.lower()
    # Remove numbers
    #doc = re.sub(r"[0-9]+", "", doc)
    # Split in tokens
    tokens = doc.split()
    # Remove Stopwords
    ###tokens = [w for w in tokens if not w in stop_words]
    # Remove punctuation
    ###tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
    # Tokens with less then two characters will be ignored
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def read_files(path):
    documents = list()
    # Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open('%s/%s' % (path, filename)) as f:
                doc = f.read()
                doc = clean_doc(doc)
                documents.append(doc)

    # Read in all lines in a txt file
    if os.path.isfile(path):
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_doc(line))
    return documents


def plot_acc_loss(title, histories, key_acc, key_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Accuracy
    ax1.set_title('Model accuracy (%s)' % title)
    names = []
    for i, model in enumerate(histories):
        ax1.plot(range(1, 7), model[key_acc])
        ax1.set_xlabel('epoch')
        names.append('Model %i' % (i+1))
        ax1.set_ylabel('accuracy')
    ax1.legend(names, loc='lower right')
    # Loss
    ax2.set_title('Model loss (%s)' % title)
    for model in histories:
        ax2.plot(range(1, 7), model[key_loss])
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
    ax2.legend(names, loc='upper right')
    fig.set_size_inches(20, 5)
    plt.show()


def show_results():
    histories = pickle.load(open('history-lstm.pkl', 'rb'))

    def get_avg(histories, his_key):
        tmp = []
        for history in histories:
            tmp.append(history[his_key][np.argmin(history['val_loss'])])
        return np.mean(tmp)

    print('Training: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'loss'),
                                                  get_avg(histories, 'acc')))
    print('Validation: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'val_loss'),
                                                    get_avg(histories, 'val_acc')))

    plot_acc_loss('training', histories, 'acc', 'loss')
    plot_acc_loss('validation', histories, 'val_acc', 'val_loss')


def main():

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NUM_WORDS)
    X_train = pad_sequences(X_train, maxlen=MAX_SEQ_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LENGTH)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    y = to_categorical(y)
    # print('Training samples: %i' % len(X))

    # docs   = negative_docs + positive_docs
    # labels = [0 for _ in range(len(negative_docs))] + [1 for _ in range(len(positive_docs))]

    # labels = to_categorical(labels)
    # print('Training samples: %i' % len(docs))

    # tokenizer.fit_on_texts(docs)
    # sequences = tokenizer.texts_to_sequences(docs)

    # word_index = tokenizer.word_index

    # result = [len(x) for x in X]
    # print('Text informations:')
    # print('max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(result),
    #                                                                                 np.min(result),
    #                                                                                 np.mean(result),
    #                                                                                 MAX_SEQ_LENGTH))
    # print('vacobulary size: %i / limit: %i' % (len(word_index), MAX_NUM_WORDS))

    # Padding all sequences to same length of `MAX_SEQ_LENGTH`
    # data = pad_sequences(X, maxlen=MAX_SEQ_LENGTH, padding='post')

    histories = []

    for i in range(RUNS):
        print('Running iteration %i/%i' % (i+1, RUNS))
        random_state = np.random.randint(1000)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, random_state=random_state)

        model = RNN(
            num_words       = MAX_NUM_WORDS,
            embedding_dim   = EMBEDDING_DIM,
            lstm_size       = LSTM_SIZE,
            lstm_layers     = LSTM_LAYERS,
            max_seq_length  = MAX_SEQ_LENGTH,
            dropout_rate    = DROPOUT_RATE,
            hidden_units    = HIDDEN_UNITS,
            nb_classes      = NB_CLASSES
        ).build_model()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        if i == 0:
            print(model.summary())
            plot_model(model, to_file='lstm_model.png', show_layer_names=False, show_shapes=True)

        history = model.fit(
            X_train, y_train,
            epochs=NB_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[
                # TQDMCallback(),
                ModelCheckpoint(
                    'model-lstm-%i.h5'%(i+1), monitor='val_loss', verbose=1, save_best_only=True, mode='min'
                ),
                # TensorBoard(log_dir='./logs/temp', write_graph=True)
            ]
        )
        print()
        histories.append(history.history)

    with open('history-lstm.pkl', 'wb') as f:
        pickle.dump(histories, f)

    show_results()


if __name__ == '__main__':
    main()
    # show_results()
