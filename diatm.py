import random
import glob
import numpy as np
from collections import defaultdict, Counter


class DiaTM:

    def __init__(self, n_topics, n_dialects, alpha=0.1):
        """Initialise the model"""

        self.n_topics = n_topics
        self.n_dialects = n_dialects
        self.alpha = alpha

    def fit(self, X):
        """Fit the model to X which is a list of collections of documents"""

        self.n_collections = len(X)

        # number of docs in all collections - sum of 1st dim of sparse matrix
        self.num_docs = sum([ collection.shape[0] for collection in X])

        # size of vocab should be in the 2nd dim of sparse matrix
        self.vocab_size = X[0].shape[1]

        self._initialize_model(X)


    def _initialize_model(self, X):
        """Prepare the model for training"""

        self.topic_counts = np.zeros(shape=(self.n_topics))
        self.collection_dialect_counts = np.zeros(shape=(self.n_collections,
                                                         self.n_dialects))

        self.topic_word_counts = np.zeros(shape=(self.n_topics,
                                                 self.vocab_size))

        self.dialect_word_counts = np.zeros(shape=(self.n_dialects,
                                                   self.vocab_size))

        self.document_topic_counts = np.zeros(shape=(self.num_docs,
                                                     self.n_topics))

        for collection in X:
            print(np.max(collection, axis=0))

        # randomly initialise state
        #random.seed()

        # randomly assign each word a topic
        #self.document_topics = np.zeros(shape=())




if __name__ == "__main__":

    from sklearn.feature_extraction.text import CountVectorizer

    doc1 = "tummy ache bad food vomit ache"
    doc4 = "vomit stomach muscle ache food poisoning"
    doc2 = "pulled muscle gym workout exercise cardio"
    doc5 = "muscle aerobic exercise cardiovascular calories"
    doc3 = "diet exercise carbs protein food health"
    doc6 = "carbohydrates diet food ketogenic protein calories"
    doc7 = "gym food gainz protein cardio muscle"
    doc8 = "stomach crunches muscle ache protein"
    doc9 = "gastroenteritis stomach vomit nausea dehydrated"
    doc10 = "dehydrated water exercise cardiovascular"
    doc11 = 'drink water daily diet health'

    # 'simple' documents
    collection1 = [doc1,doc2,doc3, doc7, doc11]
    # 'scientific' documents
    collection2 = [doc4,doc5,doc6, doc8, doc9, doc10]

    collections = [collection1, collection2]

    all_docs = collection1 + collection2

    cv = CountVectorizer()

    cv.fit(all_docs)

    X = [cv.transform(collection) for collection in collections]

    dtm = DiaTM(n_topics=3, n_dialects=2)

    dtm.fit(X)
