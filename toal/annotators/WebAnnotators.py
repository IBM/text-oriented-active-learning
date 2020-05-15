from .AbstractAnnotator import AbstractAnnotator

class WebAnnotator(AbstractAnnotator):

    def annotate(self, unlab_index, unlabeled_x, unlabeled_y):
        raise NotImplementedError()