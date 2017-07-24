import random
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
from itertools import chain

def sample_from(weights):
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random() # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w
        #print i, w, rnd
        if rnd <= 0:  # return the smallest i such that
            return i  # weigths[0] + ... + weigths[i] >= rnd


class DiaTM:

    def __init__(self, n_topics, n_dialects, alpha=0.1, beta=0.1):
        """Initialise the model"""

        self.n_topics = n_topics
        self.n_dialects = n_dialects
        self.alpha = alpha
        self.beta = beta

    def fit(self, X):
        """Fit the model to X which is a list of collections of documents"""


        self._initialize_model(X)
        self._fit(X)


    def _initialize_model(self, X):
        """Prepare the model for training"""

        ## prepare documents
        self.n_collections = len(X)

        self.vocab = Counter()

        self.all_docs = []

        for collection in X:
            self.all_docs.extend(collection)
            for doc in collection:
                self.vocab.update(doc)

        self.document_lengths = [len(x) for x in self.all_docs]

        self.W = len(self.vocab)
        self.D = sum([len(collection) for collection in X])



        self.doc_collections = [[doc in collection for
                                 collection in X].index(True)
                                for doc in self.all_docs]

        ## Initialise model by assigning everything a random state
        random.seed()

        self.topic_counts = np.zeros(shape=(self.n_topics, ))
        self.dialect_counts = np.zeros(shape=(self.n_dialects, ))
        self.collection_dialect_counts = np.zeros(shape=(len(collections),
                                                    self.n_dialects))

        self.topic_word_counts = [Counter() for i in range(0, self.n_topics)]

        self.dialect_word_counts = [Counter()
                                    for i in range(0, self.n_dialects)]

        self.document_topic_counts = np.zeros(shape=(self.D,
                                                     self.n_topics))

        self.topic_dialect_words = [[Counter() for d in range(self.n_dialects)]
                               for k in range(self.n_topics)]


        self.document_topics = [[random.randrange(self.n_topics)
                            for word in document]
                           for document in  chain(*X)]

        self.document_dialects = [[random.randrange(self.n_dialects)
                              for word in document]
                             for document in self.all_docs]


        # initialise counters
        for d, doc in enumerate(chain(*X)):
            for word, topic, dialect in zip(doc, self.document_topics[d],
                                            self.document_dialects[d]):

                self.collection_dialect_counts[self.doc_collections[d]][dialect] += 1
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.dialect_word_counts[dialect][word] += 1
                self.topic_counts[topic]+=1
                self.topic_dialect_words[topic][dialect][word] += 1


    def _fit(self, X):

        for iteration in range(0, 50):
            print("Iter {}".format(iteration))

            for d in range(self.D):
                for i, (word, topic, dialect) in enumerate(
                    zip(self.all_docs[d],
                        self.document_topics[d],
                        self.document_dialects[d])):

                    # remove current word/topic from the counts
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[d] -= 1

                    self.topic_dialect_words[topic][dialect][word] -= 1
                    self.dialect_word_counts[dialect][word] -= 1
                    self.collection_dialect_counts[self.doc_collections[d]][dialect] -= 1

                    # choose new topic based on the topic weights
                    new_topic = self.choose_new_topic(d, word)
                    self.document_topics[d][i] = new_topic

                    # add new topic back to the counts
                    self.document_topic_counts[d][new_topic] += 1


                    self.topic_word_counts[new_topic][word] += 1
                    self.topic_counts[new_topic] += 1
                    self.document_lengths[d] += 1

                    # choose new dialect based on dialect collection weights
                    new_dialect = self.choose_new_dialect(d,word)
                    self.document_dialects[d][i] = new_dialect

                    # add new dialect back to the counts
                    self.collection_dialect_counts[self.doc_collections[d]][new_dialect] += 1
                    self.dialect_word_counts[new_dialect][word] += 1


                    self.topic_dialect_words[new_topic][new_dialect][word] += 1


    def choose_new_topic(self, doc, word):
        """Given a word's topic weightings, choose a new topic"""

        return sample_from([self.topic_weight(doc, word, k)
                            for k in range(self.n_topics)])


    def p_dialect_given_collection(self, dialect, collection):
        """The probability of a dialect given a collection id"""
        return (self.collection_dialect_counts[collection][dialect]
                ) / (sum(self.collection_dialect_counts[collection]))

    def p_word_given_dialect(self, word, dialect, alpha=0.1):
        """The probability of a word occuring within a dialect"""
        #print("dialect_word_counts", dialect_word_counts[dialect][word] )
        return ((self.dialect_word_counts[dialect][word] + self.alpha) /
                (sum(self.dialect_word_counts[dialect].values()) +
                 self.W*self.alpha))

    def p_word_given_topic(self, word, topic):
        """the fraction of words assigned to topic"""

        return ((self.topic_word_counts[topic][word] + self.beta) /
                (self.topic_counts[topic] + self.W * self.beta))


    def p_topic_given_document(self, topic, d, alpha=0.1):
        """the fraction of words in d that are assigned to topic (plus some smoothing)"""

        return ((self.document_topic_counts[d][topic]+self.alpha) /
                (self.document_lengths[d]+ self.n_topics*self.alpha))

    def dialect_weight(self, d, word, dia):
        """given doc d and word, return the weight for the k-th dialect"""
        return (self.p_word_given_dialect(word, dia) *
                self.p_dialect_given_collection(dia, self.doc_collections[d]))

    def topic_weight(self, d, word, k):
        """given doc d and word, return the weight for the k-th topic"""
        return (self.p_word_given_topic(word, k) *
                self.p_topic_given_document(k, d))


    def choose_new_dialect(self, d, word):
        """Given a word's dialect weightings, choose a new dialect."""

        return sample_from([self.dialect_weight(d,word,dia)
                            for dia in range(self.n_dialects)])

if __name__ == "__main__":

    from sklearn.feature_extraction.text import CountVectorizer

    doc1 = "tummy ache bad food vomit ache"
    doc4 = "vomit stomach muscle ache food poisoning"
    doc2 = "pulled muscle gym workout exercise cardio"
    doc5 = "muscle aerobic exercise cardiovascular calories"
    doc3 = "diet exercise carbs protein food health"
    doc6 = "carbohydrates diet food ketogenic protein calories"
    doc7 = "gym food gainz tummy protein cardio muscle"
    doc8 = "stomach crunches muscle ache protein"
    doc9 = "gastroenteritis stomach vomit nausea dehydrated"
    doc10 = "dehydrated water exercise cardiovascular"
    doc11 = 'drink water daily diet health'

    # 'simple' documents
    collection1 = [doc.split(" ") for doc in
                   [doc1,doc2,doc3, doc7, doc11]]

    # 'scientific' documents
    collection2 = [doc.split(" ") for doc in
                   [doc4,doc5,doc6, doc8, doc9, doc10]]

    collections = [collection1, collection2]


    dtm = DiaTM(n_topics=3, n_dialects=2)

    dtm.fit(collections)



    topics_dict = defaultdict(lambda:[])


    for topic in range(dtm.n_topics):
        for dia in range(dtm.n_dialects):
            words = [ word if count > 0 else '' for word, count in dtm.topic_dialect_words[topic][dia].most_common()[:5] ]
            topics_dict["Topic"+str(topic) +" Dialect " + str(dia)] += words

    topics_df = pd.DataFrame.from_dict(topics_dict)

    print(topics_df)
