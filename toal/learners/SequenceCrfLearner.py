from sklearn_crfsuite import CRF
from .AbstractLearner import AbstractLearner
from toal.stores import BasicStore

class SequenceCrfLearner(AbstractLearner):

    def __init__(self):
        super().__init__()
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def train(self, store: BasicStore):
        self.clf.fit(*store.XYs)
        return self.clf
