##cython: language_level=3
##cython: boundscheck=False
##cython: wraparound=False
##cython: initializedcheck=False
##cython: cdivision=True

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


cdef double lgamma(double x) nogil:
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
    vocab_size

    cpdef public double alpha, beta, eta

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

        self.document_dialect_counts = np.zeros(shape=(self.n_documents, self.n_dialects), dtype=DTYPE)

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
            self.document_dialect_counts[doc][dia] += 1
            self.topic_word_counts[topic][word] += 1
            self.dialect_word_counts[dia][word] += 1
            self.topic_counts[topic]+=1
            self.topic_dialect_words[topic][dia][word] += 1


    def transform(self, X, max_iter=20, tol=1e-16):
        """Transform the data X according to previously fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        Note
        ----
        This uses the "iterated pseudo-counts" approach described
        in Wallach et al. (2009) and discussed in Buntine (2009).

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = utils.matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        """Transform a single document according to the previously fit model

        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document

        Note
        ----

        See Note in `transform` documentation.

        """
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1): # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc


    cpdef void fit(self, list X):
        """Fit the model to the given list of documents"""
        self._initialize(X)

        for it in range(self.n_iter):

            print("Iter:", it)
            print("LL:" +str(self._loglikelihood()))

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
        cdef long[:]  topic_counts = self.topic_counts
        cdef long[:,:,:] topic_dialect_words = self.topic_dialect_words
        cdef long[:,:] dialect_word_counts = self.dialect_word_counts
        cdef long[:,:] collection_dialect_counts = self.collection_dialect_counts

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

        # ok this is the fun bit
        #with nogil:
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
            dec(dialect_word_counts[dia,word])
            dec(collection_dialect_counts[col,dia])

            # choose new topic based on the topic weights

            dist_cum = 0

            for k in range(n_topics):

              p_word_given_topic = <double>(top_word_counts[k, word] + self.alpha) / <double>( np.sum(top_word_counts[k]) + self.alpha*self.vocab_size)

              p_topic_given_document = (<double>(doc_top_counts[doc,k] + self.alpha) /
                      <double>(doc_lens[doc]+ n_topics*self.alpha))

              for d in range(n_dialects):

                p_word_given_dialect = (dialect_word_counts[d,word] + self.alpha) / ( np.sum(dialect_word_counts[d]) + self.alpha * self.vocab_size)

                p_dialect_given_document = (<double>doc_dia_counts[doc,d] + self.eta
                        ) / <double>(cython_sum(doc_dia_counts[doc]) + self.n_documents*self.eta)

                p_dialect_given_collection = (<double>collection_dialect_counts[col,d] + self.eta
                        ) / <double>(cython_sum(collection_dialect_counts[col]) + self.n_collections*self.eta)


                p_word_topic_dialect = (topic_dialect_words[k,d,word] + self.beta) / (np.sum(topic_dialect_words[k,d]) + vocab_size*self.beta)

                dist_cum += (p_topic_given_document
                              * p_dialect_given_document
                              * p_dialect_given_collection
                              * p_word_given_dialect
                              * p_word_given_topic
                              * p_word_topic_dialect)

                #print((k*self.n_dialects)+d )
                dist_sum[ (k*self.n_dialects)+d ] = dist_cum

              #dist_cum += p_word_given_topic * p_topic_given_document
              #dist_sum_n[k] = dist_cum

            r = rands[n % n_rand] * dist_cum
            new_idx = searchsorted(dist_sum, (n_topics*n_dialects)-1, r)

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

            # add new dialect back to the counts
            inc(collection_dialect_counts[col,new_dialect])
            inc(dialect_word_counts[new_dialect,word])

            inc(topic_dialect_words[new_topic,new_dialect,word])

    cpdef double _loglikelihood(self):

        cdef int k, d, col
        cdef int D = self.document_topic_counts.shape[0]
        cdef long[:] coffset = self.collection_offsets

        cdef long[:,:] ndz = self.document_topic_counts
        cdef long[:,:] ndw = self.dialect_word_counts
        cdef long[:,:] nzw = self.topic_word_counts
        cdef long[:,:] ncd = self.collection_dialect_counts

        cdef long[:] nz = self.topic_counts
        cdef int[:] nd = np.sum(ndz,axis=1).astype(np.intc)
        cdef int[:] nn = np.sum(ncd, axis=1).astype(np.intc)
        cdef double ll = 0

        # calculate log p(w|z) and p(w|dia)
        cdef double lgamma_eta, lgamma_alpha
        with nogil:
            lgamma_eta = lgamma(self.eta)
            lgamma_alpha = lgamma(self.alpha)

            ll += self.n_topics * lgamma(self.eta * self.vocab_size)
            for k in range(max(self.n_dialects,self.n_topics)):
                ll -= lgamma(self.eta * self.vocab_size + nz[k])
                for w in range(self.vocab_size):

                  if k < self.n_topics:
                    # if nzw[k, w] == 0 addition and subtraction cancel out
                    if nzw[k, w] > 0:
                        ll += lgamma(self.eta + nzw[k, w]) - lgamma_eta

                  if k < self.n_dialects:
                    if ndw[k, w] > 0:
                        ll += lgamma(self.eta + ndw[k, w]) - lgamma_eta

            # calculate log p(z) and p(dia|c)
            for d in range(D):

                col = coffset[d]

                ll += (lgamma(self.alpha * self.n_topics) -
                        lgamma(self.alpha * self.n_topics + nd[d]))

                ll += (lgamma(self.eta * self.n_dialects) -
                        lgamma(self.eta * self.n_dialects + nn[col]))

                for k in range(max(self.n_topics, self.n_dialects)):

                    if k < self.n_topics:
                        if ndz[d, k] > 0:
                            ll += lgamma(self.alpha + ndz[d, k]) - lgamma_alpha

                    if k < self.n_dialects:
                        if ncd[col, k] > 0:
                          ll += lgamma(self.eta + ncd[col, k]) - lgamma_eta
            return ll
