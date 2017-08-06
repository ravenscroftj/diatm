#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

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

from cython.view cimport array as cvarray
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

cdef int cython_sum(long[:] y) nogil:    #changed `int` to `long`
    cdef int N = y.shape[0]
    cdef int x = y[0]
    cdef int i
    for i in xrange(1,N):
        x += y[i]
    return x


def concatenate_csr_matrices_by_row(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csr_matrix((new_data, new_indices, new_ind_ptr))


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
    _rands, \
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

    cpdef double alpha, beta, eta

    cpdef public object feature_names, docs

    def __init__(self, n_topics, n_dialects, n_iter=50, alpha=0.1, beta=0.1, eta=0.1, feature_names=None, random_state=None):
        self.n_topics = n_topics
        self.n_dialects = n_dialects
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.feature_names = feature_names

        # random numbers that are reused
        rng = utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates


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
        cdef int n

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

            print("Iter:", it)

            self._sample()

    cdef void _sample(self) except *:

        cdef int word, vocab_size, topic, dia, col, new_topic, new_dialect, n_topics, n_dialects

        # store local references to number of topics and dialects for nogil
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
        cdef long[:,:] doc_top_counts = self.document_topic_counts
        cdef long[:,:] top_word_counts = self.topic_word_counts
        cdef long[:]  topic_counts = self.topic_counts
        cdef long[:,:,:] topic_dialect_words = self.topic_dialect_words
        cdef long[:,:] dialect_word_counts = self.dialect_word_counts
        cdef long[:,:] collection_dialect_counts = self.collection_dialect_counts

        # access to the random numbers
        cdef double[:] rands = self._rands
        cdef int n_rand = self._rands.shape[0]

        # set up cumulative distribution stuff for random choices
        cdef double dist_cum = 0
        cdef double* dist_sum_n = <double*> malloc(n_topics * sizeof(double))
        cdef double* dist_sum_d = <double*> malloc(n_dialects * sizeof(double))

        #check that there was enough memory to allocate above dist vars
        if dist_sum_n is NULL or dist_sum_d is NULL:
            raise MemoryError("Could not allocate memory during sampling.")

        # N is the number of words  in all the documents
        cdef int N = WS.shape[0]

        cdef double r, p_word_given_topic, p_word_given_dialect, p_topic_given_document, p_dialect_given_collection
        cdef int n
        cdef int doc

        # ok this is the fun bit
        with nogil:
          for n in range(N):
              word = WS[n]
              doc = DS[n]
              topic = ZS[n]
              dia = NS[n]
              col = coffset[doc]

              # remove current word/topic from the counts
              dec(doc_top_counts[doc,topic])

              dec(top_word_counts[topic,word])
              dec(topic_counts[topic])
              dec(doc_lens[doc])

              dec(topic_dialect_words[topic,dia,word])
              dec(dialect_word_counts[dia,word])
              dec(collection_dialect_counts[col,dia])

              # choose new topic based on the topic weights
  #            new_topic = self.choose_new_topic(doc, word)

              dist_cum = 0

              for k in range(n_topics):

                p_word_given_topic = (<double>(top_word_counts[k,word] + self.beta) /
                        <double>(topic_counts[k] + vocab_size * self.beta))

                p_topic_given_document = (<double>(doc_top_counts[doc,k] + self.alpha) /
                        <double>(doc_lens[doc]+ n_topics*self.alpha))

                dist_cum += p_word_given_topic * p_topic_given_document
                dist_sum_n[k] = dist_cum

              r = rands[n % n_rand] * dist_cum
              new_topic = searchsorted(dist_sum_n, n_topics, r)

              ZS[n] = new_topic

              # add new topic back to the counts
              inc(doc_top_counts[doc,new_topic])

              inc(top_word_counts[new_topic,word])
              inc(topic_counts[new_topic])
              inc(doc_lens[doc])

              # choose new dialect based on dialect collection weights
  #            new_dialect = self.choose_new_dialect(doc,word)

              dist_cum = 0
              for d in range(n_dialects):

                p_word_given_dialect = ((<double>dialect_word_counts[d,word] + self.alpha) /
                        <double>(cython_sum(dialect_word_counts[d,:])+vocab_size*self.alpha))

                #print(collection_dialect_counts[col][d])
                #print(cython_sum(collection_dialect_counts[col]))



                p_dialect_given_collection = (<double>collection_dialect_counts[col,d] + self.eta
                        ) / <double>(cython_sum(collection_dialect_counts[col]) + self.n_collections*self.eta)

                #print("p_word_given_dialect is:{}".format(p_word_given_dialect))
                #print("p_dialect_given_collection is {}".format(p_dialect_given_collection))

                dist_cum += p_word_given_dialect * p_dialect_given_collection
                dist_sum_d[d] = dist_cum

                #print("Dialect prob for {} is {}".format(d, dist_cum))

              r = rands[n % n_rand] * dist_cum
              new_dialect = searchsorted(dist_sum_d, n_dialects, r)
              NS[n] = new_dialect

              # add new dialect back to the counts
              inc(collection_dialect_counts[col,new_dialect])
              inc(dialect_word_counts[new_dialect,word])

              inc(topic_dialect_words[new_topic,new_dialect,word])

        # of course we are in c land which means freeing our arrays
        free(dist_sum_n)
        free(dist_sum_d)
