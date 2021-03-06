#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import itertools
import numpy as np
cimport numpy as np

import logging
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


cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil except *:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)

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


cdef int searchsorted(double[:] arr, int length, double value) nogil:
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

logger = logging.getLogger(__name__)

cdef class DiaTM:

    cpdef public np.ndarray WS, \
    DS, \
    ZS, \
    NS, \
    _rands, \
    collection_offsets, \
    document_topic_counts, \
    document_dialect_counts, \
    topic_dialect_counts, \
    topic_word_counts, \
    topic_counts, \
    document_lengths, \
    topic_dialect_words, \
    dialect_word_counts, \
    dialect_counts, \
    collection_dialect_counts, \
    components_

    cpdef public int n_topics, \
    n_documents, \
    n_dialects, \
    n_collections, \
    n_iter, \
    vocab_size, \
    log_every

    cpdef public double alpha, beta, eta

    cpdef public object feature_names, docs

    def __init__(self, n_topics, n_dialects, n_iter=50, alpha=0.1, beta=0.1, eta=0.1, log_every=10, feature_names=None, random_state=None):
        self.n_topics = n_topics
        self.n_dialects = n_dialects
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.feature_names = feature_names
        self.log_every = log_every

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

        self.document_dialect_counts = np.zeros(shape=(self.n_documents, self.n_dialects), dtype=DTYPE)

        self.topic_dialect_words = np.zeros(shape=(self.n_topics, self.n_dialects, self.vocab_size), dtype=DTYPE)

        self.topic_dialect_counts = np.zeros(shape=(self.n_topics, self.n_dialects), dtype=DTYPE)

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
            self.document_dialect_counts[doc][dia] += 1
            self.topic_word_counts[topic][word] += 1
            self.topic_dialect_counts[topic][dia] += 1
            self.dialect_word_counts[dia][word] += 1
            self.dialect_counts[dia]+=1
            self.topic_counts[topic]+=1
            self.topic_dialect_words[topic][dia][word] += 1


    def transform(self, X):

        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)

        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = utils.matrix_to_lists(X)

        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d])
        return doc_topic

    def _transform_single(self, doc):

      alphasum = self.topic_dialect_words + self.alpha
      probs = ((self.topic_dialect_words[:,:,doc] + self.alpha) /
                alphasum.sum(axis=2)[:,:,np.newaxis])

      return probs.sum(axis=(1,2)) / probs.sum()


    cpdef void fit(self, list X):
        """Fit the model to the given list of documents"""
        self._initialize(X)

        for it in range(self.n_iter):

            if it % self.log_every == 0:
              print("Iter: {}, LL: {}".format(it,self._loglikelihood()))

            self._sample()

        self.components_ = (self.topic_word_counts + self.eta)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]



    cdef void _sample(self) except *:

        cdef int word, vocab_size, topic, dia, col, new_topic, new_dialect, \
                 new_idx, n_topics, n_dialects

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
        cdef long[:,:] doc_dia_counts = self.document_dialect_counts
        cdef long[:,:] top_word_counts = self.topic_word_counts
        cdef long[:] dialect_counts = self.dialect_counts
        cdef long[:] topic_counts = self.topic_counts
        cdef long[:,:,:] topic_dialect_words = self.topic_dialect_words
        cdef long[:,:] dialect_word_counts = self.dialect_word_counts
        cdef long[:,:] collection_dialect_counts = self.collection_dialect_counts
        cdef long[:,:] topic_dialect_counts = self.topic_dialect_counts

        # access to the random numbers
        cdef double[:] rands = self._rands
        cdef int n_rand = self._rands.shape[0]

        # set up cumulative distribution stuff for random choices
        cdef double dist_cum = 0
        req_space = n_dialects * n_topics
        cdef np.ndarray dist_sum_np =  np.zeros( req_space,dtype=np.float64 )
        cdef double[:] dist_sum = dist_sum_np


        # N is the number of words  in all the documents
        cdef int N = WS.shape[0]

        cdef double r, p_word_given_topic, p_word_given_dialect, p_topic_given_document, p_dialect_given_collection, p_word_topic_dialect
        cdef int n,k,d
        cdef int doc
        cdef double p_topic_accum, p_dia_accum
        cdef double alphasum = self.alpha*self.vocab_size
        cdef double betasum = self.beta*self.vocab_size
        cdef double etasum = self.eta*self.vocab_size

        cdef double dia_col_collection_denom, dia_given_doc_denom, word_given_dia_denom, topic_given_doc_denom

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
              dec(doc_dia_counts[doc,dia])

              dec(top_word_counts[topic,word])
              dec(topic_counts[topic])
              dec(doc_lens[doc])

              dec(topic_dialect_words[topic,dia,word])
              dec(topic_dialect_counts[topic,dia])

              dec(dialect_counts[dia])
              dec(dialect_word_counts[dia,word])
              dec(collection_dialect_counts[col,dia])

              # choose new topic based on the topic weights



              dia_given_doc_denom = <double>(cython_sum(doc_dia_counts[doc]) + self.n_documents*self.alpha)

              #pre-calculate some of the values outside of the double loop
              dia_col_collection_denom = <double>(cython_sum(collection_dialect_counts[col]) + self.n_dialects*self.eta)

              topic_given_doc_denom =  <double>( doc_lens[doc] +  self.alpha*self.n_topics)


              dist_cum = 0
              for k in range(n_topics):

                p_word_given_topic = (<double>(top_word_counts[k, word] + self.alpha) / <double>( topic_counts[k] +  alphasum))

                p_topic_given_document = (<double>(doc_top_counts[doc,k] + self.alpha) /  topic_given_doc_denom )

                p_topic_accum = p_word_given_topic * p_topic_given_document

                for d in range(n_dialects):

                  p_word_given_dialect = <double>((dialect_word_counts[d,word] + self.alpha) / ( dialect_counts[d]) + alphasum)

                  p_dialect_given_document = (<double>doc_dia_counts[doc,d] + self.alpha ) / dia_given_doc_denom

                  p_dialect_given_collection = (<double>collection_dialect_counts[col,d] + self.eta) / dia_col_collection_denom


                  p_word_topic_dialect = (topic_dialect_words[k,d,word] + self.beta) / (topic_dialect_counts[k,d] +  betasum)

                  dist_cum += (p_topic_accum * p_dialect_given_document
                                * p_dialect_given_collection
                                * p_word_given_dialect
                                * p_word_topic_dialect)

                  #print((k*self.n_dialects)+d )
                  dist_sum[ (k*self.n_dialects)+d ] = dist_cum


              r = rands[n % n_rand] * dist_cum
              new_idx = searchsorted(dist_sum, (n_topics*n_dialects), r)

              #print("New idx is {}".format(new_idx))

              new_dialect = new_idx % self.n_dialects
              new_topic = new_idx // self.n_dialects

              #print(new_dialect, new_topic)

              ZS[n] = new_topic
              NS[n] = new_dialect

              # add new topic back to the counts
              inc(doc_top_counts[doc,new_topic])
              inc(doc_dia_counts[doc,new_dialect])

              inc(top_word_counts[new_topic,word])
              inc(topic_counts[new_topic])
              inc(doc_lens[doc])

              inc(dialect_counts[new_dialect])

              # add new dialect back to the counts
              inc(collection_dialect_counts[col,new_dialect])
              inc(dialect_word_counts[new_dialect,word])

              inc(topic_dialect_words[new_topic,new_dialect,word])
              inc(topic_dialect_counts[new_topic,new_dialect])

    cpdef double _loglikelihood(self) except *:

        cdef int k, d, col
        cdef int D = self.document_topic_counts.shape[0]
        cdef long[:] coffset = self.collection_offsets

        cdef long[:,:] ndz = self.document_topic_counts
        cdef long[:,:] ndd = self.document_dialect_counts
        cdef long[:,:] ndw = self.dialect_word_counts
        cdef long[:,:] nzw = self.topic_word_counts
        cdef long[:,:] ncd = self.collection_dialect_counts
        cdef long[:,:] topic_dialect_counts = self.topic_dialect_counts
        cdef long[:,:,:] topic_dialect_words = self.topic_dialect_words

        cdef long[:] nz = self.topic_counts
        cdef long[:] dz = self.dialect_counts
        cdef int[:] dd = np.sum(ndd, axis=1).astype(np.intc)
        cdef int[:] nd = np.sum(ndz,axis=1).astype(np.intc)
        cdef int[:] nn = np.sum(ncd, axis=1).astype(np.intc)
        cdef double ll = 0

        # calculate log p(w|z) and p(w|dia)
        cdef double lgamma_eta, lgamma_alpha
        #with nogil:
        lgamma_eta = lgamma(self.eta)
        lgamma_alpha = lgamma(self.alpha)

        ll += self.n_topics * lgamma(self.eta * self.vocab_size)


        for i in range(self.n_topics * self.n_dialects):

          d = i // self.n_topics
          k = i % self.n_topics

          ll -= lgamma(self.alpha * self.vocab_size + nz[k])
          ll -= lgamma(self.eta * self.vocab_size + dz[d])
          ll -= lgamma(self.beta * self.vocab_size + topic_dialect_counts[k,d])

          for w in range(self.vocab_size):

              # if nzw[k, w] == 0 addition and subtraction cancel out
              if nzw[k, w] > 0:
                  ll += lgamma(self.alpha + nzw[k, w]) - lgamma_eta

              if ndw[d, w] > 0:
                  ll += lgamma(self.eta + ndw[d, w]) - lgamma_eta

              if topic_dialect_words[k,d,w] > 0:
                  ll += lgamma(self.beta + topic_dialect_words[k,d,w]) - lgamma_eta


        # calculate log probs for p(z), p(dia|d) and p(dia|c)
        for d in range(D):

            col = coffset[d]

            # p(topic|document)
            ll += (lgamma(self.alpha * self.n_topics) -
                    lgamma(self.alpha * self.n_topics + nd[d]))

            # p(dialect|document)
            ll += (lgamma(self.eta * self.n_dialects) -
                    lgamma(self.eta * self.n_dialects + dd[d]))

            # p(dialect|collection)
            ll += (lgamma(self.eta * self.n_dialects) -
                    lgamma(self.eta * self.n_dialects + nn[col]))

            for k in range(max(self.n_topics, self.n_dialects)):

                if k < self.n_topics:
                    if ndz[d, k] > 0:
                        ll += lgamma(self.alpha + ndz[d, k]) - lgamma_alpha

                if k < self.n_dialects:

                    if ndd[d, k] > 0:
                        ll += lgamma(self.alpha + ndd[d, k]) - lgamma_alpha


                    if ncd[col, k] > 0:
                      ll += lgamma(self.eta + ncd[col, k]) - lgamma_eta
        return ll
