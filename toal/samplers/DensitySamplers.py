import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .AbstractSampler import AbstractSampler
from toal.stores import BasicStore

# TODO need to make global constants -> or enum equivalent
# BINARY = 'binary'
# MULTICLASS = 'multiclass'
# SEQUENTIAL = 'sequential'

# ['lc', 'ms', 'ent'] -> least confidence, margin sampling, entropy


class BaseDensitySampler(AbstractSampler):

    def __init__(self, learner, strategy='ent', beta=2):
        if strategy not in ['lc', 'ms', 'ent']:
            raise ValueError("Strategy not valid, pick one of [lc, ms, ent]")
        self.strategy = strategy
        self.learner = learner
        self.beta = beta


class BinaryDensitySampler(BaseDensitySampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:

        pred_confidence = self.learner.predict_proba(store.unlabeled_Xs)

        if self.strategy == 'lc':  # least confidence selection
            raise NotImplementedError
        elif self.strategy == 'ms':  # margin sampling
            raise NotImplementedError
        elif self.strategy == 'ent':  # default to entropy
            # (negative) entropy
            prod = pred_confidence * np.ma.log(pred_confidence).filled(0)  # use mask for log(0)
            x = np.sum(prod, axis=1)
        else:
            raise ValueError("Invalid strategy")

        # TODO calculate only once and index?
        sim_mat = cosine_similarity(store.unlabeled_Xs)
        avg_sim = np.mean(sim_mat, axis=1) ** self.beta
        density_weighted = x * avg_sim

        # sorted indexes
        sorted_dw = np.argsort(density_weighted)
        rel_idx = sorted_dw[:batch_size]
        unlabeled_selection = store.unlabeled_df.iloc[rel_idx, :]

        # TODO also return the prediction confidence for debugging (and explainability)?
        return unlabeled_selection


class SequentialDensitySampler(BaseDensitySampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:

        raise NotImplementedError


class MulticlassDensitySampler(BaseDensitySampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:
        """Return the index of the chosen instances."""

        pred_confidence = np.array(self.learner.predict_proba(store.unlabeled_Xs))

        if self.strategy == 'lc':  # least confidence selection
            raise NotImplementedError
        elif self.strategy == 'ms':  # margin sampling
            raise NotImplementedError
        elif self.strategy == 'ent':  # entropy
            # (negative) entropy
            prod = pred_confidence * np.ma.log(pred_confidence).filled(0)  # use mask for log(0)
            x = np.sum(prod[:, :, 1], axis=0)
        else:
            raise ValueError("Invalid strategy")

        # TODO calculate only once and index?
        sim_mat = cosine_similarity(store.unlabeled_Xs)
        avg_sim = np.mean(sim_mat, axis=1) ** self.beta
        density_weighted = x * avg_sim

        # sorted indexes
        sort_unc = np.argsort(density_weighted)
        rel_idx = sort_unc[:batch_size]
        # return unlabeled_unit_idx[rel_idx], x[rel_idx]  # return annotation index

        unlabeled_selection = store.unlabeled_df.iloc[rel_idx, :]

        # TODO also return the prediction confidence for debugging (and explainability)?
        return unlabeled_selection


class MulticlassClusterSampler(BaseDensitySampler):

    def choose_instances(self, store: BasicStore, batch_size):

        # TODO refactor
        pred_confidence = np.array(self.learner.predict_proba(store.unlabeled_Xs))

        if self.strategy == 'lc':  # least confidence selection
            raise NotImplementedError
        elif self.strategy == 'ms':  # margin sampling
            raise NotImplementedError
        elif self.strategy == 'ent':  # entropy
            # (negative) entropy
            prod = pred_confidence * np.ma.log(pred_confidence).filled(0)  # use mask for log(0)
            x = np.sum(prod[:, :, 1], axis=0)
        else:
            raise ValueError("Invalid strategy")

        # sorted indexes
        sort_unc = np.argsort(x)

        # cluster
        kmeans = KMeans(n_clusters=batch_size, random_state=0).fit(store.unlabeled_Xs)

        # add an equal number of instances from each cluster
        # TODO this should be fixed
        inst_per_label = int(batch_size/kmeans.n_clusters)
        label_cnt = {x: 0 for x in range(kmeans.n_clusters)}
        sample_id_list = []
        for i in sort_unc:
            if label_cnt[kmeans.labels_[i]] < inst_per_label:
                sample_id_list.append(i)
                label_cnt[kmeans.labels_[i]] += 1
                if len(sample_id_list) >= batch_size:
                    break

        unlabeled_selection = store.unlabeled_df.iloc[sample_id_list, :]
        return unlabeled_selection
