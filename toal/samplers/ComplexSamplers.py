import numpy as np
import pandas as pd

from .AbstractSampler import AbstractSampler
from toal.stores import BasicStore

# TODO need to make global constants -> or enum equivalent
# BINARY = 'binary'
# MULTICLASS = 'multiclass'
# SEQUENTIAL = 'sequential'

# ['lc', 'ms', 'ent'] -> least confidence, margin sampling, entropy


class BaseComplexSampler(AbstractSampler):

    def __init__(self, learner, strategy='ent'):
        if strategy not in ['lc', 'ms', 'ent']:
            raise ValueError("Strategy not valid, pick one of [lc, ms, ent]")
        self.strategy = strategy
        self.learner = learner


class BinaryComplexSampler(BaseComplexSampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:

        raise NotImplementedError

class SequentialComplexSampler(BaseComplexSampler):

    def choose_instances(self, store: BasicStore, batch_size=10) -> pd.DataFrame:

        raise NotImplementedError


class MulticlassComplexSampler(BaseComplexSampler):

    def choose_instances(self, store: BasicStore, batch_size=10, selection='least_worst') -> pd.DataFrame:
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

        # TODO Unless we reindex unlabeled_df not sure how to match indexes of 'x' here and store.unlabeled_df

        #  aggregate Xs over annotation ids
        # ann_unit_ids = store.unlabeled_df['annotation_unit_id'].unique()

        ann_i_map = {}
        agg_annotation = {}
        for i, score in enumerate(x):
            au_id = store.unlabeled_df.iloc[i]['annotation_unit_id']
            if au_id in agg_annotation:
                agg_annotation[au_id].append(score)
                ann_i_map[au_id].append(i)
            else:
                agg_annotation[au_id] = [score]
                ann_i_map[au_id] = [i]

        if selection == 'sum':
            selection_func = lambda f: sum(f)
        elif selection == 'mean':
            selection_func = lambda f: sum(f)/len(f)
        elif selection == 'best':
            selection_func = lambda f: min(f)  # most negative is most uncertain
        elif selection == 'least_worst':
            selection_func = lambda f: max(f)
        else:
            raise ValueError("Invalid selection.")

        # sum aggregation
        sorted_agg = sorted(agg_annotation.items(), key=lambda item: selection_func(item[1]))
        texts = [x[0] for x in sorted_agg]

        # new df
        rows = list(zip(texts, texts))
        new_unlabeled_df = pd.DataFrame(sorted_agg[:batch_size], columns=['text', 'annotation_unit_id'])
        return new_unlabeled_df
