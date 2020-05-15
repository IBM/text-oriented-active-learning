import numpy as np
from sklearn import metrics
from keras.models import Sequential

# NOTE: we disabled this warning:
# UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples. 'precision', 'predicted', average, warn_for)

class F1Evaluator:

    def evaluate(self, model, x, y):

        if y is None:
            return
        test_pred = model.predict(x)

        if isinstance(model, Sequential):
            # convert matrix with one-hot rows to array with label indexes
            pred_1d = test_pred.reshape((-1, test_pred.shape[-1])).argmax(axis=1)
            gold_1d = y.reshape((-1, test_pred.shape[-1])).argmax(axis=1)
            # ignore 0 labels for evaluation
            tok_indexes = np.where(gold_1d != 0)  # 0 is the 'pad' label
            gold_1d = gold_1d[tok_indexes]
            pred_1d = pred_1d[tok_indexes]
            # Equivalent of:
            # score = metrics.f1_score(gold_1d, pred_1d, average='macro')
            _, _, score, _ = metrics.precision_recall_fscore_support(gold_1d, pred_1d, beta=1, pos_label=1, warn_for=(), average='macro')
            print(metrics.confusion_matrix(gold_1d, pred_1d))
        else:
            # Equivalent of:
            # score = metrics.f1_score(self.data_manager.test_labels, test_pred, average='macro')
            _, _, score, _ = metrics.precision_recall_fscore_support(y, test_pred, beta=1, pos_label=1, warn_for=(), average='macro')
            #print(metrics.confusion_matrix(y, test_pred))

        print("F1: %.3f" % score)

        return score
