import numpy as np
from .AbstractLearner import AbstractLearner
from toal.stores import BasicStore


class MulticlassWatsonDiscoveryLearner(AbstractLearner):

    def __init__(self, wds_resp, enrichment='relations', filter_type=None):
        # TODO different stuff I'm not checking here
        score_term = 'confidence' if enrichment == 'entities' else 'score'
        self.type_score_list = [(relation['type'], float(relation[score_term])) for result in wds_resp['results']
                                for relation in result['enriched_text'][enrichment] if filter_type is None or filter_type == relation['type']]

    def train(self, store: BasicStore):
        return None

    def predict_proba(self, unlabeled_X):
        # need to build a 3-d array of predictions for sampling: label (here type) x instances x 0/1
        # not checking that this unlabeled and WDS predictions match, but they should
        assert unlabeled_X.shape[0] == len(self.type_score_list), \
            "The unlabeled instances and predictions should be of the same length."
        labels = list(set([x[0] for x in self.type_score_list]))
        labels.sort()
        label_i_map = {label: i for i, label in enumerate(labels)}
        num_labels = len(labels)
        pred_list = [np.zeros((len(self.type_score_list), 2))] * len(labels)
        for inst_i, type_score in enumerate(self.type_score_list):
            label, score = type_score
            # pred = np.zeros((num_labels, 2))
            # pred[label_i_map[label], 1] = score
            # pred[label_i_map[label], 0] = 1 - score
            if score > 1:
                print("WARN: Score is greater than 1.")  # This shouldn't happen
            # pred_list.append(pred)
            pred_list[label_i_map[label]][inst_i, 1] = score
            pred_list[label_i_map[label]][inst_i, 0] = 1- score

        return pred_list
