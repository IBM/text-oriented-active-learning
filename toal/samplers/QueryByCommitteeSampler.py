from .AbstractSampler import AbstractSampler

class QueryByCommitteeSampler(AbstractSampler):

    def choose_instances(self, model, unlabeled, batch_size=10):
        raise NotImplementedError
