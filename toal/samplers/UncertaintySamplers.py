import numpy as np
import pandas as pd

from .AbstractSampler import AbstractSampler
from toal.stores import BasicStore

# TODO need to make global constants -> or enum equivalent
# BINARY = 'binary'
# MULTICLASS = 'multiclass'
# SEQUENTIAL = 'sequential'

# ['lc', 'ms', 'ent'] -> least confidence, margin sampling, entropy


class BaseUncertaintySampler(AbstractSampler):

    def __init__(self, learner, strategy='ent'):
        if strategy not in ['lc', 'ms', 'ent']:
            raise ValueError("Strategy not valid, pick one of [lc, ms, ent]")
        self.strategy = strategy
        self.learner = learner


class BinaryUncertaintySampler(BaseUncertaintySampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:

        pred_confidence = self.learner.predict_proba(store.unlabeled_Xs)

        if self.strategy == 'lc':  # least confidence selection
            x = np.max(pred_confidence, axis=1)
        elif self.strategy == 'ms':  # margin sampling
            x = np.abs(pred_confidence[:, 0] - pred_confidence[:, 1])
        elif self.strategy == 'ent':  # default to entropy
            # (negative) entropy
            # prod = pred_confidence * np.ma.log(pred_confidence).filled(0)  # use mask for log(0)
            prod = pred_confidence * np.log(pred_confidence)
            x = np.sum(prod, axis=1)
        else:
            raise ValueError("Invalid strategy")

        # sorted indexes
        sort_unc = np.argsort(x)
        rel_idx = sort_unc[:batch_size]
        unlabeled_selection = store.unlabeled_df.iloc[rel_idx, :]

        # TODO also return the prediction confidence for debugging (and explainability)?
        return unlabeled_selection


class SequentialUncertaintySampler(BaseUncertaintySampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:

        pred_confidence = self.learner.predict_proba(store.unlabeled_Xs)

        if self.strategy == 'lc':  # least confidence selection
            raise NotImplementedError
        elif self.strategy == 'ms':  # margin sampling
            raise NotImplementedError
        elif self.strategy == 'ent':  # default to entropy
            # (negative) entropy
            prod = pred_confidence * np.ma.log(pred_confidence).filled(0)  # use mask for log(0)
            # using token entropy or total token entropy (see Settles and Craven, EMNLP'08) for the moment
            x = np.sum(np.sum(prod, axis=2) * (store.unlabeled_Xs > 0).astype(int), axis=1)  # only sum where there is actually a token
        else:
            raise ValueError("Invalid strategy")

        # sorted indexes
        sort_unc = np.argsort(x)
        rel_idx = sort_unc[:batch_size]
        unlabeled_selection = store.unlabeled_df.iloc[rel_idx, :]
        return unlabeled_selection


class MulticlassUncertaintySampler(BaseUncertaintySampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:
        """Return the index of the chosen instances."""

        pred_confidence = np.array(self.learner.predict_proba(store.unlabeled_Xs))

        if self.strategy == 'lc':  # least confidence selection
            x = np.max(pred_confidence[:, :, 1], axis=0)
        elif self.strategy == 'ms':  # margin sampling
            # ap = np.argpartition(pred_confidence, -2, axis=1)[:, -2:]
            # top2_preds = pred_confidence[np.arange(pred_confidence.shape[0])[:, None], ap]
            # x = np.abs(top2_preds[:, 0] - top2_preds[:, 1])
            raise NotImplementedError
        elif self.strategy == 'ent':  # entropy
            # (negative) entropy
            prod = pred_confidence * np.ma.log(pred_confidence).filled(0)  # use mask for log(0)
            x = np.sum(prod[:, :, 1], axis=0)
        else:
            raise ValueError("Invalid strategy")

        # sorted indexes
        sort_unc = np.argsort(x)
        rel_idx = sort_unc[:batch_size]
        # return unlabeled_unit_idx[rel_idx], x[rel_idx]  # return annotation index

        unlabeled_selection = store.unlabeled_df.iloc[rel_idx, :]

        # TODO also return the prediction confidence for debugging (and explainability)?
        return unlabeled_selection
