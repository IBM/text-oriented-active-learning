from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from .AbstractLearner import AbstractLearner
from toal.stores import BasicStore

class MulticlassLearner(AbstractLearner):

    def __init__(self):
        super().__init__()
        clf = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial')
        # clf = RandomForestClassifier(n_estimators=10)
        # clf = RidgeClassifier(alpha=0.3)
        # clf = MLPClassifier()
        self.clf = MultiOutputClassifier(clf, n_jobs=-1)

    def train(self, store: BasicStore):
        # x = self.data_manager.train
        # y = self.data_manager.train_labels
        self.clf.fit(*store.train_XYs)
        return self.clf

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

