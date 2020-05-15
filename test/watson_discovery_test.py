import unittest
import json
from toal.stores import BasicStore
from toal.stores.loaders import load_from_watson_discover, load_w_annotation_units_from_watson_discover
from toal.learners import MulticlassWatsonDiscoveryLearner
from toal.samplers import MulticlassUncertaintySampler, MulticlassComplexSampler, MulticlassClusterSampler


class PandasStoreTest(unittest.TestCase):

    def test_watson_service(self):

        with open('../data/is_promoted.json') as json_f:
            wds_resp = json.load(json_f)

        # Using default encoders, same of writing BasicStore(extractor=TfIdfExtractor(), encoder=GenericLabelBinarizer())
        store = BasicStore()
        store.append_data( *load_from_watson_discover(wds_resp, 'relations'))

        learner = MulticlassWatsonDiscoveryLearner(wds_resp, 'relations')
        # We are going to call it
        # model = learner.train(store)

        # sampler = MulticlassUncertaintySampler(learner, strategy='lc')
        sampler = MulticlassClusterSampler(learner)
        sampled_df = sampler.choose_instances(store, batch_size=10)

        # can't retrain WKS
        # and we don't have gold data in WDS, so we can't evaluate (e.g., measure F1)

        # should do qualitative evaluation
        print(sampled_df)

    def test_watson_service_complex_annotation(self):
        toi = "is_promoted"  # type of interest  is_acquiring  transacted is_promoted
        query = "enriched_text.relations.type:" + toi
        enrichment = 'relations'

        with open('../data/is_promoted.json') as json_f:
            wds_resp = json.load(json_f)

        # Using default encoders, same of writing BasicStore(extractor=TfIdfExtractor(), encoder=GenericLabelBinarizer())
        store = BasicStore()
        store.append_data( *load_w_annotation_units_from_watson_discover(wds_resp, enrichment=enrichment, filter_type=toi))

        learner = MulticlassWatsonDiscoveryLearner(wds_resp, filter_type=toi)
        # We are going to call it
        # model = learner.train(store)

        sampler = MulticlassComplexSampler(learner)
        sampled_df = sampler.choose_instances(store, batch_size=50, selection='least_worst')

        # can't retrain WKS
        # and we don't have gold data in WDS, so we can't evaluate (e.g., measure F1)

        # should do qualitative evaluation
        print(sampled_df)
