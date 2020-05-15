from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from .BaseEncoder import BaseEncoder


# This is a copy and paste of tfidf_extractor.py with some code moving, testing for the moment
class SequenceEncoder(BaseEncoder):

    def __init__(self, max_sent_len=60):
        self.__max_sent_len = max_sent_len
        self.__tokenizer = Tokenizer(filters='')

    # Just a logical way to split Xs and Ys extraction
    def encode(self, labeled_delta_df=None):

        super().encode(labeled_delta_df)

        labels = self._store.labeled_df['label']

        self.__tokenizer.fit_on_texts(labels)
        self.refit()

    def refit(self):

        labels = self._store.labeled_df['label']

        y_sequences = self.__tokenizer.texts_to_sequences(labels)

        y_seq = pad_sequences(y_sequences, maxlen=self.__max_sent_len)
        y_oh_seq = to_categorical(y_seq)

        self._y = y_oh_seq
        self._index_to_label_map = self.__tokenizer.index_word
