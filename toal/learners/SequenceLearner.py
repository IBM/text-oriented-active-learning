from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Activation, TimeDistributed
from .AbstractLearner import AbstractLearner
from .support import load_w2v_embeddings
from toal.stores import BasicStore

__max_vocab_size__ = 30000
__emb_len__ = 300


class SequenceLearner(AbstractLearner):

    clf: Sequential = None

    # TODO: Context is weird here too
    def __init__(self, store: BasicStore, use_embeddings=True):

        super().__init__()

        num_labels = len(store.available_labels) + 1  # extra empty label for zero-padding
        vocab_size = len(store.word_to_index_map)
        self.epoch_cnt = 0

        # prepare embedding matrix
        num_words = min(__max_vocab_size__, vocab_size) + 1  # extra 0 for zero-padding

        try:
            # get pretrained embedding
            if use_embeddings:
                embedding_weights = load_w2v_embeddings(store.word_to_index_map, __max_vocab_size__)
            else:
                embedding_weights = None
        except FileNotFoundError:
            embedding_weights = None

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    __emb_len__,
                                    weights=embedding_weights,
                                    mask_zero=True,
                                    trainable=True)

        self.clf = Sequential()
        self.clf.add(embedding_layer)
        self.clf.add(LSTM(32, return_sequences=True))
        self.clf.add(Dropout(0.5))
        self.clf.add(TimeDistributed(Dense(num_labels)))
        self.clf.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        self.clf.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        # self.clf.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.clf.summary()  # for debugging network graph

    def train(self, store: BasicStore):

        batch_size = 32
        epochs = self.epoch_cnt + 2

        # if True:  # TODO add some if online_learning switch
        #     x = self.data_manager.train_batch
        #     y = self.data_manager.train_batch_labels
        #     batch_size = 10
        # else:
        #     x = self.data_manager.train
        #     y = self.data_manager.train_labels
        self.clf.fit(
            store.labeled_Xs,
            store.Ys,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            initial_epoch=self.epoch_cnt+1,
            shuffle=True
        )

        self.epoch_cnt = epochs
        return self.clf

    def predict_proba(self, X):
        return self.clf.predict(X)
