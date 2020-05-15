import unittest
from toal.stores import BasicStore
from toal.stores.loaders import load_single_label_csv, load_multi_label_csv

class PandasStoreTest(unittest.TestCase):

    def test_load_train_set_from_csv(self):

        store = BasicStore()
        labeled_df, unlabeled_df = load_single_label_csv("../data/obligation-data.csv", split_type="train")
        store.append_data(labeled_df, unlabeled_df)

        self.assertEqual(len(store.labeled_df.index), len(labeled_df.index))
        self.assertEqual(len(store.train_df.index), len(labeled_df.index))
        self.assertEqual(len(store.test_df.index), 0)
        self.assertEqual(unlabeled_df, None)
        self.assertEqual(len(store.unlabeled_df.index), 0)

        # Call method on the dataframe to check they works
        store.labeled_df.head()
        store.train_df.head()
        store.test_df.head()
        store.unlabeled_df.head()


    def test_load_multilabel_trainset_from_csv(self):

        store = BasicStore()
        labeled_df, unlabeled_df = load_multi_label_csv("../data/colo_NYS_data_train_6_underscores.csv", split_type="test")
        store.append_data(labeled_df, unlabeled_df)

        self.assertEqual(len(store.labeled_df.index), len(labeled_df.index))
        self.assertEqual(len(store.test_df.index), len(labeled_df.index))
        self.assertEqual(len(store.train_df.index), 0)
        self.assertEqual(unlabeled_df, None)
        self.assertEqual(len(store.unlabeled_df.index), 0)

        # Call method on the dataframe to check they works
        store.labeled_df.head()
        store.train_df.head()
        store.test_df.head()
        store.unlabeled_df.head()
