import abc
import pandas as pd
from toal.stores import BasicStore

class BaseEncoder(abc.ABC):

    _store: BasicStore = None
    _index_to_label_map: dict = None
    _y = None

    def assign_store(self, store: BasicStore):
        if store       is     None: raise ValueError("Store cannot be none")
        if self._store is not None: raise ValueError("Store already set, impossible to set again")
        self._store = store
        return self

    @property
    def Ys(self):
        return self._y

    @property
    def index_to_label_map(self) -> dict:
        return self._index_to_label_map

    @abc.abstractmethod
    def encode(self, labeled_delta_df=None):
        """
        Method called from the store everytime there is an update
        :param labeled_delta: The new labeled data received
        :param unlabeled_delta:  The new unlabeled data received
        """
        if self._store is None:
            raise ValueError("Store not initialized, cannot encode.")

    # def filter_Ys(self, filter: pd.Series):
    #     self._y = self._y[filter]