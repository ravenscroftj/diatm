import random
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from itertools import chain


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

        self.document_topics = np.random.choice(self.n_topics, (self.n_documents, longest_doc))
        self.document_dialects = np.random.choice(self.n_dialects, (self.n_documents, longest_doc))

        self.document_lengths = self.docs.sum(axis=1)


        # initialise counters
        for d in range(self.docs.shape[0]):

            doc_words = chain(*[[word] * self.docs[d,word]
                                          for (_,word) in np.transpose(self.docs[d].nonzero()) ])

            for word, topic, dialect in zip(doc_words, self.document_topics[d],
                                            self.document_dialects[d]):

                self.collection_dialect_counts[self.collection_offsets[d]][dialect] += 1
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.dialect_word_counts[dialect][word] += 1
                self.topic_counts[topic]+=1
                self.topic_dialect_words[topic][dialect][word] += 1

    def fit(self, X):
        self._initialize(X)


        for it in range(self.n_iter):

            print("Iter:", it)

            for d in range(self.n_documents):


                doc_words = chain(*[[word] * self.docs[d,word]
                                          for (_,word) in np.transpose(self.docs[d].nonzero()) ])

                for i, (word, topic, dialect) in enumerate(
                    zip(doc_words,
                        self.document_topics[d],
                        self.document_dialects[d])):


                    # remove current word/topic from the counts
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[d] -= 1

                    self.topic_dialect_words[topic][dialect][word] -= 1
                    self.dialect_word_counts[dialect][word] -= 1
                    self.collection_dialect_counts[self.collection_offsets[d]][dialect] -= 1

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
                    self.collection_dialect_counts[self.collection_offsets[d]][new_dialect] += 1
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
