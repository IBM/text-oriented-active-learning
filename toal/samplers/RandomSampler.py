from random import shuffle
import pandas as pd
from .AbstractSampler import AbstractSampler
from toal.stores import BasicStore


class RandomSampler(AbstractSampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:
        # just get the next n (random) samples
        shuf_idx = list(range(store.unlabeled_Xs.shape[0]))
        shuffle(shuf_idx)

        unlabeled_selection = store.unlabeled_df.iloc[shuf_idx[:batch_size], :]

        # TODO also return the prediction confidence for debugging (and explainability)?
        return unlabeled_selection

