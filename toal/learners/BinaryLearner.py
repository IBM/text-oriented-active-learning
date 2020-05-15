from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from .AbstractLearner import AbstractLearner
from toal.stores import BasicStore

class BinaryLearner(AbstractLearner):

    def __init__(self):
        super().__init__()
        # self.clf = LinearSVC(C=0.3)
        self.clf = LogisticRegression(C=1)

    def train(self, store: BasicStore):
        self.clf.fit(*store.train_XYs)
        return self.clf

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

