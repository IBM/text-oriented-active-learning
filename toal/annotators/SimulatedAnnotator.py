import pandas as pd
from .AbstractAnnotator import AbstractAnnotator


# Note, probably it should be a "yield from" function, or something asynchronous anyway,
# to allow WebAnnotators to return result in a async fashion.
class SimulatedAnnotator(AbstractAnnotator):

    def __init__(self, text_label_map):
        self.text_label_map = text_label_map

    def annotate(self, unlabeled_df: pd.DataFrame, labels: list) -> pd.DataFrame:

        annotations_df = unlabeled_df.copy()

        annotations_df["train"] = True
        annotations_df["label"] = annotations_df['text'].apply(lambda x: self.text_label_map[x])

        return annotations_df
