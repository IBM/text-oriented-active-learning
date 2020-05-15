import abc
from toal.stores import BasicStore

class AbstractLearner(abc.ABC):

    @abc.abstractmethod
    def train(self, store: BasicStore):
        """Train your model."""
        raise NotImplementedError

    # @classmethod
    # def __subclasshook__(cls, C):
    #     if cls is Learner:
    #         if any("train" in B.__dict__ for B in C.__mro__):
    #             return True
    #     return NotImplemented