from sklearn.feature_extraction.text import TfidfVectorizer

from .BaseExtractor import BaseExtractor

class TfIdfExtractor(BaseExtractor):

    _labeled_x = None
    _unlabeled_x = None

    __vectorizer = None

    # "delta" is not used at the moment, it can be used in the future for optimization (e.g. caching,
    # only diff computation, etc...)
    def extract(self, labeled_delta=None, unlabeled_delta=None):

        super().extract(labeled_delta, unlabeled_delta)

        # feature extraction with Tf-Idf weights for word features
        # get feature matrix X
        # TODO: this guy should be in the constructor, probably
        self.__vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.95)

        labeled_text   = self._store.labeled_df  ['text']
        unlabeled_text = self._store.unlabeled_df['text']

        all_text = labeled_text.append(unlabeled_text)

        # fit to all data
        self.__vectorizer.fit( all_text )

        self.refit(labeled_delta, unlabeled_delta)

        self._word_to_index_map = self.__vectorizer.vocabulary_
        self._index_to_word_map = { i: w for w, i in self.__vectorizer.vocabulary_.items() }


    # "delta" is not used at the moment, it can be used in the future for optimization (e.g. caching,
    # only diff computation, etc...)
    def refit(self, labeled_delta=None, unlabeled_delta=None):

        super().extract(labeled_delta, unlabeled_delta)

        labeled_text   = self._store.labeled_df  ['text']
        unlabeled_text = self._store.unlabeled_df['text']

        if len(labeled_text) > 0:
            self._labeled_x   = self.__vectorizer.transform(labeled_text)

        if len(unlabeled_text) > 0:
            self._unlabeled_x = self.__vectorizer.transform(unlabeled_text)

        if len(labeled_text) + len(unlabeled_text) == 0:
            print("No data to extract features from!")
            raise ValueError("No data to extract features from!")

        # If I want an X with both:
        # self.__x = vectorizer.transform(all_text)

