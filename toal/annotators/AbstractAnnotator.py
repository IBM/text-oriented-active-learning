import abc

class AbstractAnnotator(abc.ABC):

    @abc.abstractmethod
    def annotate(self, unlab_index, unlabeled):
        raise NotImplementedError()