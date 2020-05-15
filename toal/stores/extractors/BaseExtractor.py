import abc
from toal.stores import BasicStore

class BaseExtractor(abc.ABC):

    # Store with the original data
    _store: BasicStore = None

    _labeled_x = None
    _unlabeled_x = None
    _word_to_index_map = None
    _index_to_word_map = None

    def assign_store(self, store: BasicStore):
        if store       is     None: raise ValueError("Store cannot be none")
        if self._store is not None: raise ValueError("Store already set, impossible to set again")
        self._store = store
        return self

    @property
    def labeled_Xs(self):
        return self._labeled_x;

    @property
    def unlabeled_Xs(self):
        return self._unlabeled_x;

    @property
    def word_to_index_map(self):
        return self._word_to_index_map

    @abc.abstractmethod
    def extract(self, labeled_delta=None, unlabeled_delta=None):
        """
        Method called from the store everytime there is an update
        :param labeled_delta: The new labeled data received
        :param unlabeled_delta:  The new unlabeled data received
        """
        if self._store is None:
            raise ValueError("Store not initialized, cannot encode.")

    @abc.abstractmethod
    def refit(self, labeled_delta=None, unlabeled_delta=None):
        """
        Method called from the store everytime there is an update
        :param labeled_delta: The new labeled data received
        :param unlabeled_delta:  The new unlabeled data received
        """
        if self._store is None:
            raise ValueError("Store not initialized, cannot encode.")
