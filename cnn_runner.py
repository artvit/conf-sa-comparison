from cnn import CNN
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


MAX_NUM_WORDS   = 15000
EMBEDDING_DIM   = 300
MAX_SEQ_LENGTH  = 500
USE_GLOVE       = False
FILTER_SIZES    = [3, 4, 5]
FEATURE_MAPS    = [200, 200, 200]
DROPOUT_RATE    = 0.4
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

# def create_glove_embeddings():
#     print('Pretrained embeddings GloVe is loading...')
#
#     embeddings_index = {}
#     f = open('glove.6B.%id.txt' % EMBEDDING_DIM)
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#     print('Found %s word vectors in GloVe embedding' % len(embeddings_index))
#
#     embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
#
#     for word, i in tokenizer.word_index.items():
#         if i >= MAX_NUM_WORDS:
#             continue
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#
#     return Embedding(
#         input_dim=MAX_NUM_WORDS,
#         output_dim=EMBEDDING_DIM,
#         input_length=MAX_SEQ_LENGTH,
#         weights=[embedding_matrix],
#         trainable=True,
#         name="word_embedding"
#     )


def plot_acc_loss(title, histories, key_acc, key_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Accuracy
    ax1.set_title('Model accuracy (%s)' % title)
    names = []
    for i, model in enumerate(histories):
        ax1.plot(model[key_acc])
        ax1.set_xlabel('epoch')
        names.append('Model %i' % (i+1))
        ax1.set_ylabel('accuracy')
    ax1.legend(names, loc='lower right')
    # Loss
    ax2.set_title('Model loss (%s)' % title)
    for model in histories:
        ax2.plot(model[key_loss])
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
    ax2.legend(names, loc='upper right')
    fig.set_size_inches(20, 5)
    plt.show()


def main():

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NUM_WORDS)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    labels = to_categorical(y)
    print('Training samples: %i' % len(X))

    # docs   = negative_docs + positive_docs
    # labels = [0 for _ in range(len(negative_docs))] + [1 for _ in range(len(positive_docs))]

    # labels = to_categorical(labels)
    # print('Training samples: %i' % len(docs))

    # tokenizer.fit_on_texts(docs)
    # sequences = tokenizer.texts_to_sequences(docs)

    # word_index = tokenizer.word_index

    result = [len(x) for x in X]
    print('Text informations:')
    print('max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(result),
                                                                                    np.min(result),
                                                                                    np.mean(result),
                                                                                    MAX_SEQ_LENGTH))
    # print('vacobulary size: %i / limit: %i' % (len(word_index), MAX_NUM_WORDS))

    # Padding all sequences to same length of `MAX_SEQ_LENGTH`
    data = pad_sequences(X, maxlen=MAX_SEQ_LENGTH, padding='post')

    histories = []

    for i in range(RUNS):
        print('Running iteration %i/%i' % (i+1, RUNS))
        random_state = np.random.randint(1000)

        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=VAL_SIZE, random_state=random_state)

        emb_layer = None
        # if USE_GLOVE:
        #     emb_layer = create_glove_embeddings()

        model = CNN(
            embedding_layer = emb_layer,
            num_words       = MAX_NUM_WORDS,
            embedding_dim   = EMBEDDING_DIM,
            filter_sizes    = FILTER_SIZES,
            feature_maps    = FEATURE_MAPS,
            max_seq_length  = MAX_SEQ_LENGTH,
            dropout_rate    = DROPOUT_RATE,
            hidden_units    = HIDDEN_UNITS,
            nb_classes      = NB_CLASSES
        ).build_model()

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adam(),
            metrics=['accuracy']
        )

        if i == 0:
            print(model.summary())

        history = model.fit(
            X_train, y_train,
            epochs=NB_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[
                # TQDMCallback(),
                ModelCheckpoint(
                    'model-%i.h5'%(i+1), monitor='val_loss', verbose=1, save_best_only=True, mode='min'
                ),
                # TensorBoard(log_dir='./logs/temp', write_graph=True)
            ]
        )
        print()
        histories.append(history.history)

    with open('history.pkl', 'wb') as f:
        pickle.dump(histories, f)
    histories = pickle.load(open('history.pkl', 'rb'))

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


if __name__ == '__main__':
    main()
