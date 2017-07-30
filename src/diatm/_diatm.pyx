from collections import Counter

import itertools
import numpy as np
cimport numpy as np



cdef class DiaTM:

  cdef int n_topics, n_dialects, n_collections, n_documents, vocab_size

  cdef np.ndarray collection_offsets

  def __init__(self, n_topics, n_dialects):
    self.n_topics = n_topics
    self.n_dialects = n_dialects



  cdef void _initialize(self, X):
    """Set up data structures for diatm model"""
    self.n_collections = len(X)

    self.n_documents = sum(collection.shape[0] for collection in X)

    self.collection_offsets = np.zeros(size=(self.n_documents), dtype=np.int)

    last_offset = 0
    for i, collection in enumerate(X):
      self.collection_offsets[last_offset:last_offset+len(collection)] = i
      last_offset += len(collection)


    self.n_documents = sum(collection.shape[0] for collection in X)
    self.vocab_size = X[0].shape[1]

    print(self.collection_offsets)






  def fit(self, X):
    self._initialize(X)
