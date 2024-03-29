from keras.layers import Activation, Input, Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, LSTMCell, CuDNNLSTM
from keras.layers.convolutional import SeparableConv1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import initializers, Sequential
from keras import backend as K


class RNN:

    __version__ = '0.0.2'

    def __init__(self, embedding_layer=None, num_words=None, embedding_dim=None,
                 max_seq_length=100, lstm_size=256, lstm_layers=1,
                 hidden_units=100, dropout_rate=None, nb_classes=None):
        """
        Arguments:
            embedding_layer : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            num_words       : Maximal amount of words in the vocabulary (default: None)
            embedding_dim   : Dimension of word representation (default: None)
            max_seq_length  : Max length of sequence (default: 100)
            filter_sizes    : An array of filter sizes per channel (default: [3,4,5])
            feature_maps    : Defines the feature maps per channel (default: [100,100,100])
            hidden_units    : Hidden units per convolution channel (default: 100)
            dropout_rate    : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes      : Number of classes which can be predicted
        """
        self.embedding_layer = embedding_layer
        self.num_words       = num_words
        self.max_seq_length  = max_seq_length
        self.embedding_dim   = embedding_dim
        self.lstm_size       = lstm_size
        self.lstm_layers     = lstm_layers
        self.hidden_units    = hidden_units
        self.dropout_rate    = dropout_rate
        self.nb_classes      = nb_classes

    def build_model(self):
        """
        Build the model

        Returns:
            Model           : Keras model instance
        """

        # Checks
        if not self.embedding_layer and (not self.num_words or not self.embedding_dim):
            raise Exception('Please define `num_words` and `embedding_dim` if you not use a pre-trained embeddings')

        model = Sequential()
        # Building embeddings from scratch
        input = Input(shape=(self.max_seq_length,), dtype='int32', name='word_input')
        if self.embedding_layer is None:
            self.embedding_layer = Embedding(
                self.num_words,
                self.embedding_dim,
                input_length=self.max_seq_length,
                name="word_embedding"
            )
        x = self.embedding_layer(input)
        # model.add(self.embedding_layer)
        # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        x = self.build_lstm(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        output = Dense(self.nb_classes, activation='softmax')(x)
        # model.add(Dense(self.nb_classes, activation='softmax'))
        return Model(input, output)

    def build_lstm(self, x):
        for _ in range(self.lstm_layers - 1):
            x = self.get_lstm_layer()(x)
            x = Dropout(self.dropout_rate)(x)
        x = self.get_lstm_layer(True)(x)
        x = Dropout(self.dropout_rate)(x)
        return x

    def get_lstm_layer(self, last=False):
        return CuDNNLSTM(self.lstm_size,
                         # dropout=self.dropout_rate,
                         # recurrent_dropout=self.dropout_rate,
                         return_sequences=not last)
