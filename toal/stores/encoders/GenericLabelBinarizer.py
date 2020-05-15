from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from .BaseEncoder import BaseEncoder

# This is a copy and paste of tfidf_extractor.py with some code moving, testing for the moment
class GenericLabelBinarizer(BaseEncoder):

    _is_multiclass = None
    _is_multilabel = None
    _encoder = None

    def encode(self, labeled_delta_df=None):

        super().encode(labeled_delta_df)

        # TODO: Work only on the delta rather then starting from scratch every time
        labels_column = self._store.labeled_df['label']

        unique_labels = set()

        is_multilabel = False
        # Old code, it was doing a full linear scan if it was no multilabel,
        # it's now done during the unique labels computations
        # is_multilabel = any(' ' in l for l in labels)

        # either multiclass or multilabel, this will work anyway
        for label in labels_column:
            splitten_labels = label.split()
            if len(splitten_labels) > 1:
                is_multilabel = True
                for inner_l in splitten_labels: unique_labels.add(inner_l)
            else:
                unique_labels.add(label)

        is_multiclass = len(unique_labels) > 2

        # multiclass classification
        if is_multiclass:
            encoder = MultiLabelBinarizer()
            if is_multilabel:
                y = encoder.fit_transform([ label.split() for label in labels_column ])
            else:
                y = encoder.fit_transform(labels_column)

        # binary classification, cannot be neither multilabel or multiclass
        else:
            encoder = LabelBinarizer()
            y = encoder.fit_transform(labels_column)

        self._encoder = encoder
        self._is_multiclass = is_multiclass
        self._is_multilabel = is_multilabel
        self._index_to_label_map = { i: l for i, l in enumerate(encoder.classes_) }
        self._y = y

    def refit(self):

        labels_column = self._store.labeled_df['label']

        # multiclass classification
        if self._is_multiclass:
            encoder = MultiLabelBinarizer()
            if self._is_multilabel:
                self._y = self._encoder.transform([ label.split() for label in labels_column ])
            else:
                self._y = self._encoder.transform(labels_column)

        # binary classification, cannot be neither multilabel or multiclass
        else:
            encoder = LabelBinarizer()
            self._y = self._encoder.transform(labels_column)
