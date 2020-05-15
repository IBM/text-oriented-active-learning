from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from .BaseExtractor import BaseExtractor

class WordEmbeddingExtractor(BaseExtractor):

    def __init__(self, max_sent_len=60):
        self.__max_sent_len = max_sent_len
        # tokenize words and then use word embedding features
        self.__tokenizer = Tokenizer(filters='')  # empty filters to match labels

    # "delta" is not used at the moment, it can be used in the future for optimization (e.g. caching,
    # only diff computation, etc...)
    def extract(self, labeled_delta=None, unlabeled_delta=None):
        # feature extraction for DNN sequence labeling

        super().extract(labeled_delta, unlabeled_delta)

        # prepare data for input X
        labeled_text   = self._store.labeled_df  ['text']
        unlabeled_text = self._store.unlabeled_df['text']
        all_text = labeled_text.append(unlabeled_text)

        self.__tokenizer.fit_on_texts(all_text)

        self.refit(labeled_delta, unlabeled_delta)

        self._word_to_index_map = self.__tokenizer.word_index
        self._index_to_word_map = self.__tokenizer.index_word

    # "delta" is not used at the moment, it can be used in the future for optimization (e.g. caching,
    # only diff computation, etc...)
    def refit(self, labeled_delta=None, unlabeled_delta=None):

        super().extract(labeled_delta, unlabeled_delta)

        labeled_text   = self._store.labeled_df  ['text']
        unlabeled_text = self._store.unlabeled_df['text']

        if len(labeled_text) > 0:
            sequences = self.__tokenizer.texts_to_sequences(labeled_text)
            self._labeled_x = pad_sequences(sequences, maxlen=self.__max_sent_len)

        if len(unlabeled_text) > 0:
            sequences = self.__tokenizer.texts_to_sequences(unlabeled_text)
            self._unlabeled_x = pad_sequences(sequences, maxlen=self.__max_sent_len)

        if len(labeled_text) + len(unlabeled_text) == 0:
            print("No data to extract features from!")
            raise ValueError("No data to extract features from!")

