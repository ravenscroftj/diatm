import random
import glob
import timeit
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from itertools import chain
from lda import utils




def concatenate_csr_matrices_by_row(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csr_matrix((new_data, new_indices, new_ind_ptr))

def sample_from(weights):
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random()  # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w
        #print i, w, rnd
        if rnd <= 0:  # return the smallest i such that
            return i  # weigths[0] + ... + weigths[i] >= rnd

class DiaTM:

    def __init__(self, n_topics, n_dialects, n_iter=50, alpha=0.1, beta=0.1, feature_names=None):
        self.n_topics = n_topics
        self.n_dialects = n_dialects
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.feature_names = feature_names


    def _initialize(self, X):
        """Set up data structures for diatm model"""

        print("initializing")
        self.n_collections = len(X)

        self.n_documents = sum(collection.shape[0] for collection in X)

        self.collection_offsets = np.zeros(shape=(self.n_documents), dtype=np.int)

        last_offset = 0
        for i, collection in enumerate(X):
            self.collection_offsets[last_offset:last_offset+collection.shape[0]] = i
            last_offset += collection.shape[0]

        self.n_documents = sum(collection.shape[0] for collection in X)
        self.vocab_size = X[0].shape[1]

        longest_doc = 0
        self.docs = X[0]
        for collection in X[1:]:
            self.docs = concatenate_csr_matrices_by_row(self.docs, collection)
            longest_doc = max(longest_doc, collection.sum(axis=1).max())

        ## Initialise model by assigning everything a random state
        random.seed()

        self.topic_counts = np.zeros(shape=(self.n_topics, ))
        self.dialect_counts = np.zeros(shape=(self.n_dialects, ))

        self.collection_dialect_counts = np.zeros(shape=(len(X),
                                                    self.n_dialects))

        self.topic_word_counts = np.zeros(shape=(self.n_topics, self.vocab_size))

        self.dialect_word_counts = np.zeros(shape=(self.n_dialects, self.vocab_size))

        self.document_topic_counts = np.zeros(shape=(self.n_documents, self.n_topics))

        self.topic_dialect_words = np.zeros(shape=(self.n_topics, self.n_dialects, self.vocab_size))

        self.document_lengths = self.docs.sum(axis=1)

        self.WS, self.DS = utils.matrix_to_lists(self.docs)

        # topic selection for word
        self.ZS = np.random.choice(self.n_topics, self.WS.shape)
        # dialect selection for word
        self.NS = np.random.choice(self.n_dialects, self.WS.shape)

        # initialise counters
        N = self.docs.sum()
        for n in range(N):

            word = self.WS[n]
            doc = self.DS[n]
            topic = self.ZS[n]
            dia = self.NS[n]
            col = self.collection_offsets[doc]

            self.collection_dialect_counts[col][dia] += 1
            self.document_topic_counts[doc][topic] += 1
            self.topic_word_counts[topic][word] += 1
            self.dialect_word_counts[dia][word] += 1
            self.topic_counts[topic]+=1
            self.topic_dialect_words[topic][dia][word] += 1

    def fit(self, X):
        self._initialize(X)

        N = self.docs.sum()
        for it in range(self.n_iter):

            #print("Iter:", it)


            for n in range(N):
                word = self.WS[n]
                doc = self.DS[n]
                topic = self.ZS[n]
                dia = self.NS[n]
                col = self.collection_offsets[doc]

                # remove current word/topic from the counts
                self.document_topic_counts[doc][topic] -= 1
                self.topic_word_counts[topic][word] -= 1
                self.topic_counts[topic] -= 1
                self.document_lengths[doc] -= 1

                self.topic_dialect_words[topic][dia][word] -= 1
                self.dialect_word_counts[dia][word] -= 1
                self.collection_dialect_counts[col][dia] -= 1

                # choose new topic based on the topic weights
                new_topic = self.choose_new_topic(doc, word)
                self.ZS[n] = new_topic

                # add new topic back to the counts
                self.document_topic_counts[doc][new_topic] += 1

                self.topic_word_counts[new_topic][word] += 1
                self.topic_counts[new_topic] += 1
                self.document_lengths[doc] += 1

                # choose new dialect based on dialect collection weights
                new_dialect = self.choose_new_dialect(doc,word)
                self.NS[n] = new_dialect

                # add new dialect back to the counts
                self.collection_dialect_counts[col][new_dialect] += 1
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

        return ((self.dialect_word_counts[dialect][word] + self.alpha) /
                (self.dialect_word_counts[dialect].sum() +
                 self.vocab_size*self.alpha))

    def p_word_given_topic(self, word, topic):
        """the fraction of words assigned to topic"""

        return ((self.topic_word_counts[topic][word] + self.beta) /
                (self.topic_counts[topic] + self.vocab_size * self.beta))


    def p_topic_given_document(self, topic, d, alpha=0.1):
        """the fraction of words in d that are assigned to topic (plus some smoothing)"""

        return ((self.document_topic_counts[d][topic]+self.alpha) /
                (self.document_lengths[d]+ self.n_topics*self.alpha))

    def dialect_weight(self, d, word, dia):
        """given doc d and word, return the weight for the k-th dialect"""
        return (self.p_word_given_dialect(word, dia) *
                self.p_dialect_given_collection(dia, self.collection_offsets[d]))

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
    import pyximport; pyximport.install()

    import _diatm

    doc0 = "stomach muscle ache food poisoning vomit nausea"
    doc1 = "muscle aerobic exercise cardiovascular calories"
    doc2 = "carbohydrates diet food ketogenic protein calories"
    doc3 = "stomach crunches muscle ache protein"
    doc4 = "gastroenteritis stomach vomit nausea dehydrated"
    doc5 = "dehydrated water exercise cardiovascular"


    # 'scientific' documents
    collection1 = [
        "stomach muscle ache food poisoning vomit nausea",
        "muscle aerobic exercise cardiovascular calories",
        "carbohydrates diet food ketogenic protein calories",
        "stomach crunches muscle ache protein",
        "gastroenteritis stomach vomit nausea dehydrated",
        "dehydrated water exercise cardiovascular"
        ]

    # 'simple' documents
    collection2 = [
        "tummy ache bad food poisoning sick",
        "pulled muscle gym workout exercise cardio",
        "diet exercise carbs protein food health",
        "gym food gainz protein cardio muscle",
        "drink water daily diet health",
    ]

    cv = CountVectorizer()

    cv.fit(collection1 + collection2)


    cm1 = cv.transform(collection1)
    cm2 = cv.transform(collection2)

    dtm = _diatm.DiaTM(n_topics=3,
                       n_dialects=2,
                       n_iter=20,
                       alpha=0.01,
                       beta=0.1,
                       eta=0.01,
                       log_every=1,
                       feature_names=cv.get_feature_names())
    #dtm = DiaTM(n_topics=3, n_dialects=2, feature_names=cv.get_feature_names())


    start = timeit.default_timer()

    dtm.fit([cm1,cm2])

    end = timeit.default_timer() - start

    print("Took {:f} seconds".format(end))

    topics_dict = defaultdict(lambda:[])

    for topic in range(dtm.n_topics):
        for dia in range(dtm.n_dialects):

            word_idx = sorted(dtm.topic_dialect_words[topic][dia].nonzero()[0],
                           key=lambda x: dtm.topic_dialect_words[topic][dia][x],
                           reverse=True)[:5]

            words = [dtm.feature_names[x] for x in word_idx]

            while len(words) < 5:
                words.append("")

            topics_dict["Topic"+str(topic) +" Dialect " + str(dia)] += words

    topics_df = pd.DataFrame.from_dict(topics_dict)

    print(topics_df)
