from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Activation
from .AbstractLearner import AbstractLearner
from .support import load_w2v_embeddings
from toal.stores import BasicStore

__max_vocab_size__ = 20000
__emb_len__ = 300
__batch_size__ = 32

class MulticlassLstmLearner(AbstractLearner):

    clf: Sequential = None

    # TODO: Check context here, it's weird
    def __init__(self, store: BasicStore):

        super().__init__()

        num_labels = len( store.index_to_label_map )
        print (f"Found {num_labels} labels.")
        vocab_size = len( store.word_to_index_map )
        print (f"Found {vocab_size} words.")

        self.epoch_cnt = 0

        # prepare embedding matrix
        num_words = min(__max_vocab_size__, vocab_size)

        try:
            # get pretrained embedding
            embedding_weights = load_w2v_embeddings(store.word_to_index_map, __max_vocab_size__)
            # embedding_weights = None
        except FileNotFoundError:
            print("[!] Skipping embedding weights, no data available.")
            embedding_weights = None

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(
            num_words,
            __emb_len__,
            weights=embedding_weights,
            mask_zero=True,
            trainable=False
        )

        self.clf = Sequential()
        self.clf.add(embedding_layer)
        self.clf.add(LSTM(32))
        self.clf.add(Dropout(0.5))
        self.clf.add(Dense(num_labels))
        self.clf.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        self.clf.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        self.clf.summary()  # for debugging network graph

    def train(self, store: BasicStore):

        epochs = self.epoch_cnt + 3

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
            batch_size=__batch_size__,
            epochs=epochs,
            validation_split=0.2,
            initial_epoch=self.epoch_cnt + 1,
            shuffle=True
         )

        self.epoch_cnt = epochs

        return self.clf