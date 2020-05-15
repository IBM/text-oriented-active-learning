import abc
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from .. import BasicStore

# This is a copy and paste of tfidf_extractor.py with some code moving, testing for the moment
class CachedEncoder(abc.ABC):

    # Store with the original data
    __store: BasicStore = None

    # Unique labels (strings)
    _unique_labels = set()

    # If text could have multiple labels associated
    _is_multilabel = False

    def __init__(self, store: BasicStore):
        if store is None:
            raise ValueError("Missing mandatory parameter store")
        self.__store = store

    @abc.abstractmethod
    def concrete_encode(self, labeled_delta_df):
        raise NotImplementedError

    def encode(self, labeled_delta_df):
        raise NotImplementedError("NOT READY YET.")
        """
        Encode the labels only if new ones are passed, otherwise just keep the old encoding.

        :param labeled_delta_df: new dataframe which came in input
        :return:   True if the encoding has been updated (new labels came in), False otherwise
        """

        new_unique_labels = set()

        # either multiclass or multilabel, this will work anyway
        for labels in labeled_delta_df['label']:
            for label in labels:
                for inner_l in label.split():
                    if len(inner_l) > 1:
                        self._is_multilabel = True
                    new_unique_labels.add(inner_l)

        if len(new_unique_labels - self._unique_labels) > 0:
            self._unique_labels = self._unique_labels + new_unique_labels
            return self.concrete_encode(label)

        return False



