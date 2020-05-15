import pandas as pd
import numpy as np
from .extractors import TfIdfExtractor
from .encoders import GenericLabelBinarizer
from typing import List, Tuple
from .extractors import BaseExtractor
from .encoders import BaseEncoder

def check_data_alignment(labeled_df, unlabeled_df):

    if labeled_df is None and unlabeled_df is None:
        raise ValueError("Labeled and unlabeled data cannot be both None.")

    for column in ["annotation_unit_id", "text", "label", "train"]:
        if labeled_df is not None and column not in labeled_df:
            print(f"Missing required column {column} in the labeled dataframe.")
            raise ValueError("Received labeled data doesn't contain required columns.")

    for column in ["annotation_unit_id", "text"]:
        if unlabeled_df is not None and column not in unlabeled_df:
            print(f"Missing required column {column} in the unlabeled dataframe.")
            raise ValueError("Received unlabeled data doesn't contain required columns.")


class BasicStore():

    __labeled_df = None
    __unlabeled_df = None

    __extractor: BaseExtractor = None
    __encoder: BaseEncoder     = None

    def __init__(self, extractor: BaseExtractor=None, encoder: BaseEncoder=None):

        super().__init__()

        self.__labeled_df   = pd.DataFrame(columns=[ 'annotation_unit_id', 'text', 'label', 'train' ])
        self.__unlabeled_df = pd.DataFrame(columns=[ 'annotation_unit_id', 'text'                   ])

        # Cannot do as a default parameter, otherwise same instance would be shared
        # across different stores.
        if extractor is None: extractor = TfIdfExtractor()
        if encoder   is None: encoder   = GenericLabelBinarizer()

        self.__extractor = extractor.assign_store(self)
        self.__encoder   = encoder  .assign_store(self)

    def append_data(self, labeled_df=None, unlabeled_df=None):

        check_data_alignment(labeled_df, unlabeled_df)

        if labeled_df is not None:
            self.__labeled_df = self.__labeled_df.append(labeled_df, ignore_index=True, sort=False)

        if unlabeled_df is not None:
            self.__unlabeled_df = self.__unlabeled_df.append(unlabeled_df, ignore_index=True, sort=False)

        # Send the delta to the extractor, so it can decide what to do
        self.__extractor.extract(labeled_delta=labeled_df, unlabeled_delta=unlabeled_df)

        # Ony call re-encode if there is new labeled data.
        if labeled_df is not None:
            self.__encoder.encode(labeled_delta_df=labeled_df)


    def update_with_annotation(self, labeled_delta: pd.DataFrame):
        """Move annotations from the unlabeled dataframe to the train."""

        # Add the new labels to the labeled dataframe
        self.__labeled_df = self.__labeled_df.append(labeled_delta, ignore_index=True, sort=False)

        # Find the filter of data to be now removed from the unlabeled set
        filter = ~self.__unlabeled_df['text'].isin(labeled_delta['text'])

        # Filter it out from the unlabeled dataset
        self.__unlabeled_df = self.__unlabeled_df[filter]

        self.__extractor.refit()
        self.__encoder.refit()


    @property
    def train_df(self):
        return self.__labeled_df[ self.__labeled_df['train'] == True ]

    @property
    def test_df(self):
        return self.__labeled_df[ self.__labeled_df['train'] == False ]

    @property
    def labeled_df(self):
        return self.__labeled_df

    @property
    def unlabeled_df(self):
        return self.__unlabeled_df

    @property
    def labeled_Xs(self) -> List[int]:
        return self.__extractor.labeled_Xs

    @property
    def unlabeled_Xs(self) -> List[int]:
        return self.__extractor.unlabeled_Xs

    @property
    def Ys(self) -> List[int]:
        return self.__encoder.Ys

    @property
    def XYs(self) -> Tuple[List[int], List[int]]:
        return (self.labeled_Xs, self.Ys)


    @property
    def word_to_index_map(self):
        return self.__extractor.word_to_index_map

    @property
    def index_to_label_map(self) -> dict:
        return self.__encoder.index_to_label_map

    @property
    def available_labels(self) -> list:
        return list(self.index_to_label_map.values())


    # TODO: performance could be improved
    # train_filter could be replaced with the index position or other more efficient techniques
    # (e.g. having two dataframe directly). We are avoiding index since at the moment we don't
    # want to assume that index are coherent (e.g. for 10 elements, index is 0...9, that could
    # not be the case sometime). This is future work, but probably the easiest is to have two
    # dataframes directly (anyway, that's how stuff look externally), with the caveaut of merging
    # them on labeled stuff.
    def _filter_XYs(self, train: bool):
        train_filter = self.__labeled_df['train'] == train
        # We have to ravel: https://stackoverflow.com/questions/29778035/scipy-sparse-csr-matrix-row-filtering-how-to-properly-achieve-it
        to_keep_train = np.ravel(np.array(train_filter))
        filtered_Xs = self.labeled_Xs[to_keep_train, :]
        filtered_Ys = self.Ys[train_filter]
        return filtered_Xs, filtered_Ys

    @property
    def train_XYs(self) -> Tuple[List[int], List[int]]:
        Xs = self._filter_XYs(True)
        return Xs

    @property
    def test_XYs(self) -> Tuple[List[int], List[int]]:
        Xs = self._filter_XYs(False)
        return Xs




    # def get_data(self, split_type=None):
    #
    #     if split_type not in [None, "train", "test", "unlabeled"]:
    #         raise ValueError(f"split_type must be None or one of 'train', 'test', or 'unlabeled', received {split_type}")
    #
    #     if split_type is None:
    #         return self.__df
    #     else:
    #         return self.__df[ self.__df['split'] == split_type ]

    # def get_annotation_unit_ids(self, split_type):
    #     df = self.get_data(split_type=split_type)
    #     return df['annotation_unit_id'].unique()

    # def get_labeled_data(self):
    #     annotation_ids = set(self.get_annotation_unit_ids('label'))
    #     instance_idx = self.__df.loc[ self.__df['annotation_unit_id' ].isin( annotation_ids )].index
    #     return self.__Xs[instance_idx, :], self.__Ys[instance_idx, :]
    #
    # def get_test_data(self):
    #     instance_idx = self.df.loc[self.df['annotation_unit_id'].isin(self.test_idx)].index
    #     return self.features[instance_idx, :], self.labels[instance_idx, :]
    #
    # def get_unlabeled_data(self):
    #     instance_idx = self.df.loc[self.df['annotation_unit_id'].isin(self.unlabeled_unit_idx)].index
    #     return self.features[instance_idx, :], instance_idx