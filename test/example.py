import unittest
import pandas as pd
from toal.stores import BasicStore
from toal.stores.loaders import load_split_multi_label_csv, load_single_label_csv, load_multi_label_csv, load_conll_2003, load_split_conll_2003
from toal.learners import BinaryLearner, MulticlassLearner, SequenceLearner
from toal.evaluators import F1Evaluator
from toal.samplers import BinaryUncertaintySampler, MulticlassUncertaintySampler, RandomSampler, SequentialUncertaintySampler
from toal.annotators import SimulatedAnnotator
from toal.stores.extractors import WordEmbeddingExtractor, SequenceExtractor
from toal.stores.encoders import SequenceEncoder
from toal.learners import MulticlassLstmLearner

class PandasStoreTest(unittest.TestCase):

    def test_binary(self):

        # Using default encoders, same of writing BasicStore(extractor=TfIdfExtractor(), encoder=GenericLabelBinarizer())
        store = BasicStore()
        labeled_df, unlabeled_df, unlabeled_map = load_split_multi_label_csv('../data/mr.csv', shuffle=True)
        store.append_data( labeled_df, unlabeled_df )

        learner = BinaryLearner()
        sampler = BinaryUncertaintySampler(learner, strategy='ent')

        annotator = SimulatedAnnotator(unlabeled_map)

        model = learner.train(store)

        f1evaluator = F1Evaluator()
        f1 = f1evaluator.evaluate(model, *store.test_XYs)
        print(f"Multiclass learner F1 evaluation score is {f1}")

        for i in range(10):
            # sample and annotate new data
            unlabeled_selection = sampler.choose_instances(store)
            annotated_df = annotator.annotate(unlabeled_selection, store.available_labels)
            store.update_with_annotation(annotated_df)

            # retrain with newly labeled data
            model = learner.train(store)

            # evaluate
            f1 = f1evaluator.evaluate(model, *store.test_XYs)
            print(f"Multiclass learner F1 evaluation score is {f1}")


    def test_multiclass(self):

        # Using default encoders, same of writing BasicStore(extractor=TfIdfExtractor(), encoder=GenericLabelBinarizer())
        store = BasicStore()
        labeled_df, unlabeled_df, unlabeled_map = load_split_multi_label_csv('../data/colo_NYS_data_train_6_underscores.csv')
        store.append_data( labeled_df, unlabeled_df )

        learner = MulticlassLearner()
        sampler = MulticlassUncertaintySampler(learner)

        annotator = SimulatedAnnotator(unlabeled_map)

        model = learner.train(store)

        f1evaluator = F1Evaluator()
        f1 = f1evaluator.evaluate(model, *store.test_XYs)
        print(f"Multiclass learner F1 evaluation score is {f1}")
        # keep track of performance per number of training instances
        # self.f1_per_instance = {labeled_x.shape[0]: f1}

        for i in range(10):
            # sample and annotate new data
            unlabeled_selection = sampler.choose_instances(store)
            annotated_df = annotator.annotate(unlabeled_selection, store.available_labels)
            store.update_with_annotation(annotated_df)

            # retrain with newly labeled data
            model = learner.train(store)

            # evaluate
            f1 = f1evaluator.evaluate(model, *store.test_XYs)
            print(f"Multiclass learner F1 evaluation score is {f1}")


    def test_sequence(self):

        store = BasicStore(extractor=SequenceExtractor(), encoder=SequenceEncoder())
        labeled_df, unlabeled_df, unlabeled_map = load_split_conll_2003('../data/conll-2003/eng.train', test_n=0, labeled_pct=0.4)
        store.append_data(labeled_df, unlabeled_df)
        labeled_df, unlabeled_df = load_conll_2003('../data/conll-2003/eng.train', 'test')
        store.append_data(labeled_df, unlabeled_df)

        learner = MulticlassLearner()
        sampler = MulticlassUncertaintySampler(learner)

        annotator = SimulatedAnnotator(unlabeled_map)

        model = learner.train(store)

        f1evaluator = F1Evaluator()
        f1 = f1evaluator.evaluate(model, *store.test_XYs)
        print(f"Multiclass learner F1 evaluation score is {f1}")
        # keep track of performance per number of training instances
        # self.f1_per_instance = {labeled_x.shape[0]: f1}

        for i in range(10):
            # sample and annotate new data
            unlabeled_selection = sampler.choose_instances(store)
            annotated_df = annotator.annotate(unlabeled_selection, store.available_labels)
            store.update_with_annotation(annotated_df)

            # retrain with newly labeled data
            model = learner.train(store)

            # evaluate
            f1 = f1evaluator.evaluate(model, *store.test_XYs)
            print(f"Multiclass learner F1 evaluation score is {f1}")

    def test_lstm_sequence(self):

        store = BasicStore(extractor=WordEmbeddingExtractor(), encoder=SequenceEncoder())
        labeled_df, unlabeled_df, unlabeled_map = load_split_conll_2003('../data/conll-2003/eng.train', test_n=0, labeled_pct=0.4)
        store.append_data(labeled_df, unlabeled_df)
        labeled_df, unlabeled_df = load_conll_2003('../data/conll-2003/eng.testa', 'test')
        store.append_data(labeled_df, unlabeled_df)

        learner = SequenceLearner(store, use_embeddings=False)
        sampler = SequentialUncertaintySampler(learner)

        # chosen_idx = chooser.choose_instances(model)
        annotator = SimulatedAnnotator(unlabeled_map)

        model = learner.train(store)

        f1evaluator = F1Evaluator()
        f1 = f1evaluator.evaluate(model, *store.test_XYs)
        # keep track of performance per number of training instances
        # self.f1_per_instance = {labeled_x.shape[0]: f1}

        print(f"LSTM sequence learner F1 evaluation score is {f1}")

        for i in range(10):
            # sample and annotate new data
            unlabeled_selection = sampler.choose_instances(store)
            annotated_df = annotator.annotate(unlabeled_selection, store.available_labels)
            store.update_with_annotation(annotated_df)

            # retrain with newly labeled data
            model = learner.train(store)

            # evaluate
            f1 = f1evaluator.evaluate(model, *store.test_XYs)
            print(f"Multiclass learner F1 evaluation score is {f1}")

    def test_wordembedding(self):

        store = BasicStore(extractor=WordEmbeddingExtractor(), encoder=SequenceEncoder())
        store.append_data( *load_multi_label_csv('../data/colo_NYS_data_train_6_underscores.csv', split_type="train") )

        learner = MulticlassLstmLearner(store)
        sampler = MulticlassUncertaintySampler(learner)

        # chosen_idx = chooser.choose_instances(model)
        annotator = SimulatedAnnotator()

        model = learner.train(store)

        f1evaluator = F1Evaluator()
        f1 = f1evaluator.evaluate(model, *store.XYs)
        # keep track of performance per number of training instances
        # self.f1_per_instance = {labeled_x.shape[0]: f1}

        print(f"Lstm multiclass learner F1 evaluation score is {f1}")
