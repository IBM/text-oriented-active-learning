import abc
from toal.stores import BasicStore

class AbstractSampler(abc.ABC):

    @abc.abstractmethod
    def choose_instances(self, store: BasicStore, batch_size):
        raise NotImplementedError

    # @classmethod
    # def __subclasshook__(cls, C):
    #     if cls is Learner:
    #         if any("train" in B.__dict__ for B in C.__mro__):
    #             return True
    #     return NotImplemented