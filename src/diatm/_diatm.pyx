from collections import Counter

import itertools
import numpy as np
cimport numpy as np

import random
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from itertools import chain
from lda import utils

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t


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


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


cdef class DiaTM:

    cpdef public np.ndarray WS, \
    DS, \
    ZS, \
    NS, \
    collection_offsets, \
    document_topic_counts, \
    topic_word_counts, \
    topic_counts, \
    document_lengths, \
    topic_dialect_words, \
    dialect_word_counts, \
    dialect_counts, \
    collection_dialect_counts

    cpdef public int n_topics, \
    n_documents, \
    n_dialects, \
    n_collections, \
    n_iter, \
    vocab_size

    cpdef double alpha, beta

    cpdef public object feature_names, docs

    def __init__(self, n_topics, n_dialects, n_iter=50, alpha=0.1, beta=0.1, feature_names=None):
        self.n_topics = n_topics
        self.n_dialects = n_dialects
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.feature_names = feature_names


    cdef void _initialize(self, X) except *:
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

        self.topic_counts = np.zeros(shape=(self.n_topics, ), dtype=DTYPE)
        self.dialect_counts = np.zeros(shape=(self.n_dialects, ), dtype=DTYPE)

        self.collection_dialect_counts = np.zeros(shape=(len(X),
                                                    self.n_dialects), dtype=DTYPE)

        self.topic_word_counts = np.zeros(shape=(self.n_topics, self.vocab_size), dtype=DTYPE)

        self.dialect_word_counts = np.zeros(shape=(self.n_dialects, self.vocab_size), dtype=DTYPE)

        self.document_topic_counts = np.zeros(shape=(self.n_documents, self.n_topics), dtype=DTYPE)

        self.topic_dialect_words = np.zeros(shape=(self.n_topics, self.n_dialects, self.vocab_size), dtype=DTYPE)

        self.document_lengths = np.empty(shape=(self.n_documents, ), dtype=DTYPE) #self.docs.sum(axis=1)
        self.document_lengths[:] = self.docs.sum(axis=1).reshape(-1)

        self.WS, self.DS = utils.matrix_to_lists(self.docs)

        # topic selection for word
        self.ZS = np.random.choice(self.n_topics, (self.WS.shape[0],))
        # dialect selection for word
        self.NS = np.random.choice(self.n_dialects, (self.WS.shape[0],))

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


    cpdef void fit(self, list X):

        self._initialize(X)

        for it in range(self.n_iter):

            #print("Iter:", it)

            self._sample()

    cdef void _sample(self) except *:

        cdef int word, vocab_size, doc, topic, dia, col, new_topic, n_topics, n_dialects

        cdef double alpha, beta

        # store local references to number of topics and dialects for nogil
        beta = self.beta
        alpha = self.alpha
        n_topics = self.n_topics
        n_dialects = self.n_dialects
        vocab_size = self.vocab_size

        # declare efficient memory views onto the numpy arrays
        cdef long[:] doc_lens = self.document_lengths
        cdef int[:] WS = self.WS
        cdef int[:] DS = self.DS
        cdef long[:] ZS = self.ZS
        cdef long[:] NS = self.NS
        cdef long[:] coffset = self.collection_offsets
        cdef long[:,:]  doc_top_counts= self.document_topic_counts
        cdef long[:,:]  top_word_counts = self.topic_word_counts
        cdef long[:]  topic_counts = self.topic_counts
        cdef long[:,:,:] topic_dialect_words = self.topic_dialect_words
        cdef long[:,:] dialect_word_counts = self.dialect_word_counts
        cdef long[:,:] collection_dialect_counts = self.collection_dialect_counts

        # set up cumulative distribution stuff for random choices
        cdef double dist_cum = 0
        cdef double* dist_sum_n = <double*> malloc(n_topics * sizeof(double))
        cdef double* dist_sum_d = <double*> malloc(n_dialects * sizeof(double))

        #check that there was enough memory to allocate above dist vars
        if dist_sum_n is NULL or dist_sum_d is NULL:
            raise MemoryError("Could not allocate memory during sampling.")

        # N is the number of words  in all the documents
        cdef int N = self.docs.sum()

        # ok this is the fun bit

        for n in range(N):
            word = WS[n]
            doc = DS[n]
            topic = ZS[n]
            dia = NS[n]
            col = coffset[doc]

            # remove current word/topic from the counts
            dec(doc_top_counts[doc][topic])

            dec(top_word_counts[topic][word])
            dec(topic_counts[topic])
            dec(doc_lens[doc])

            dec(topic_dialect_words[topic][dia][word])
            dec(dialect_word_counts[dia][word])
            dec(collection_dialect_counts[col][dia])

            # choose new topic based on the topic weights
            new_topic = self.choose_new_topic(doc, word)

            dist_cum = 0

            for k in range(n_topics):

              p_word_given_topic = ((top_word_counts[topic][word] + beta) /
                      (topic_counts[topic] + vocab_size * beta))

              p_topic_given_document = ((doc_top_counts[doc][topic] + alpha) /
                      (doc_lens[doc]+ n_topics*alpha))

              dist_cum += p_word_given_topic * p_topic_given_document
              dist_sum_n[k] = dist_cum

            #r = rands[i % n_rand] * dist_cum

            #new_topic = searchsorted(dist_sum, n_topics, r)


            ZS[n] = new_topic

            # add new topic back to the counts
            inc(doc_top_counts[doc][new_topic])

            inc(top_word_counts[new_topic][word])
            inc(topic_counts[new_topic])
            inc(doc_lens[doc])

            # choose new dialect based on dialect collection weights
            new_dialect = self.choose_new_dialect(doc,word)

            self.NS[n] = new_dialect

            # add new dialect back to the counts
            inc(collection_dialect_counts[col][new_dialect])
            inc(dialect_word_counts[new_dialect][word])

            inc(topic_dialect_words[new_topic][new_dialect][word])

        # of course we are in c land which means freeing our arrays
        free(dist_sum_n)
        free(dist_sum_d)

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
    collection1 = [doc1,doc2,doc3, doc7, doc11]

    # 'scientific' documents
    collection2 = [doc4,doc5,doc6, doc8, doc9, doc10]

    cv = CountVectorizer()

    cv.fit(collection1 + collection2)


    cm1 = cv.transform(collection1)
    cm2 = cv.transform(collection2)

    dtm = DiaTM(n_topics=3, n_dialects=2, feature_names=cv.get_feature_names())

    dtm.fit([cm1,cm2])



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
