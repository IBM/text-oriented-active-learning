########################################
#            TO BE DELETED             #
########################################

from .AbstractLearner import AbstractLearner
from sklearn_crfsuite import CRF


class CrfLearner(AbstractLearner):

    def __init__(self):
        self.clf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def train(self, x, y):
        self.clf.fit(x, y)
        return self.clf

