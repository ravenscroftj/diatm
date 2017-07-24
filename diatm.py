import random
import glob
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix


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

        self.dialect_counts = np.zeros(shape=self.n_dialects)

        self.collection_dialect_counts = np.zeros(shape=(self.n_collections,
                                                         self.n_dialects))

        self.topic_word_counts = np.zeros(shape=(self.n_topics,
                                                 self.vocab_size))

        self.dialect_word_counts = np.zeros(shape=(self.n_dialects,
                                                   self.vocab_size))

        self.document_topic_counts = np.zeros(shape=(self.num_docs,
                                                     self.n_topics))


        longest_doc = max([collection.shape[1] for collection in X])

        self.document_topics = lil_matrix((self.num_docs, longest_doc),
                                     dtype=np.int8)

        self.document_dialects = lil_matrix((self.num_docs, longest_doc),
                                     dtype=np.int8)

        self.topic_dialect_words = np.zeros(shape=(self.topic_counts,
                                                   self.dialect_counts,
                                                   self.vocab_size))

        doc_offset = 0

        self.document_lengths = np.zeros(shape(self.num_docs,))

        for c, collection in enumerate(X):

            # find total word count
            N = collection.sum()
            #generate a random topic and dialect for every word
            t_rands = np.random.choice(self.n_topics, N)
            d_rands = np.random.choice(self.n_dialects, N)

            for doc, word in np.transpose(collection.nonzero()):

                for i, topic, dialect in zip(range(1, collection[doc, word]+1),
                                             t_rands,
                                             d_rands):

                    self.document_topics[doc + doc_offset, word] = topic
                    self.document_dialects[doc+doc_offset, word] = dialect
                    self.document_topic_counts[doc+doc_offset, topic] += 1

                    self.collection_dialect_counts[c, dialect] += 1

                    self.topic_word_counts[topic][word] += 1
                    self.dialect_word_counts[dialect][word] += 1

                    self.topic_counts[topic] += 1
                    self.dialect_counts[dialect] += 1

                    self.topic_dialect_words[topic][dialect][word] += 1


            # once we get to here, increase the doc_offset by collection length
            doc_offset += collection.shape[0]


    def _fit(self, X):

        doc_offset = 0

        for c, collection in enumerate(X):

            for doc, word in np.transpose(collection.nonzero()):

                for i in range(1, collection[doc, word]+1):

                    topic = self.document_topics[doc+doc_offset, word]
                    dialect = self.document_dialects[doc+doc_offset, word]

                    self.document_topic_counts[doc+doc_offset, topic] -= 1
                    self.topic_word_counts[topic, word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[doc+doc_offset] -= 1

                    self.topic_dialect_words[topic, dialect, word] -= 1
                    self.dialect_word_counts[dialect, word] -= 1
                    self.collection_dialect_counts[c, dialect] -= 1

                    # randomly choose new topic and dialect
                    new_topic = choose_new_topic(doc, word)

                    # add new topic back to the counts
                    self.document_topic_counts[doc, new_topic] += 1

                    self.topic_word_counts[new_topic, word] += 1
                    self.topic_counts[new_topic] += 1
                    self.document_lengths[doc] += 1

                    # choose new dialect based on dialect collection weights
                    new_dialect = choose_new_dialect(d,word)
                    self.document_dialects[doc, i] = new_dialect

                    # add new dialect back to the counts
                    collection_dialect_counts[doc_collections[d]][new_dialect] += 1
                    dialect_word_counts[new_dialect][word] += 1


                    topic_dialect_words[new_topic][new_dialect][word] += 1

            # once we get to here, increase the doc_offset by collection length
            doc_offset += collection.shape[0]


    def choose_new_topic(self, doc, word):
        """Given a word's topic weightings, choose a new topic"""

        return sample_from([self.topic_weight(doc, word, k)
                            for k in range(self.n_topics)])


    def p_dialect_given_collection(dialect, collection):
        """The probability of a dialect given a collection id"""
        return (collection_dialect_counts[collection][dialect]) / (sum(collection_dialect_counts[collection]))

    def p_word_given_dialect(word, dialect, alpha=0.1):
        """The probability of a word occuring within a dialect"""
        #print("dialect_word_counts", dialect_word_counts[dialect][word] )
        return ( (dialect_word_counts[dialect][word] + alpha) /
                (sum(dialect_word_counts[dialect].values()) + W*alpha) )

    def p_word_given_topic(word, topic):
        """the fraction of words assigned to topic"""

        return ((self.topic_word_counts[topic, word] + self.beta) /
                (self.topic_counts[topic] + self.vocab_size * self.beta))


    def p_topic_given_document(topic, d, alpha=0.1):
        """the fraction of words in d that are assigned to topic (plus some smoothing)"""

        return ((document_topic_counts[d][topic]+alpha) /
                (document_lengths[d]+ n_topics*alpha))

    def dialect_weight(d, word, dia):
        """given doc d and word, return the weight for the k-th dialect"""
        return p_word_given_dialect(word, dia) * p_dialect_given_collection(dia, doc_collections[d])

    def topic_weight(d, word, k):
        """given doc d and word, return the weight for the k-th topic"""
        return (self.p_word_given_topic(word, k) *
                self.p_topic_given_document(k, d))

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
