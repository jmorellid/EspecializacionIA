import numpy as np

corpus = np.array(['hola como estas', 'todo bien vos gatito', 'el domingo hubo mucho sol', 'ser o no ser'])

class One_Hot_Enconding(object):

    def __init__(self):
        self.corpus = None
        self.word_set = np.array([])

    def fit(self, corpus):

        self.corpus = corpus

        word_list = np.array([])

        for text in corpus:
            words = text.lower().split(' ')
            word_list = np.append(word_list, words)

        word_set = np.unique(word_list)

        self.word_set = word_set

        return word_set


    def transform(self, corpus = None):
        
        word_set = self.word_set
        one_hot_array = np.zeros(shape=[self.corpus.shape[0], word_set.shape[0]] )
        
        if corpus is None:
             corpus = self.corpus

        for i, text in enumerate(corpus):
            words = np.array(text.lower().split(' '))
            one_hot_array[i,:] = np.isin(word_set, words)

        return one_hot_array


    def fit_transform(self, corpus):
        self.corpus = corpus

        one_hot_array = np.array([])
        word_list = np.array([])

        word_set = self.fit(self, corpus)
        one_hot_array = self.transform(self, corpus)

        return one_hot_array


class ValueFrequency(object):

    def __init__(self):
        self.corpus = None
        self.word_set = np.array([])

    def fit(self, corpus):
        self.corpus = corpus

        word_list = np.array([])

        for text in corpus:
            words = text.lower().split(' ')
            word_list = np.append(word_list, words)

        word_set = np.unique(word_list)

        self.word_set = word_set

        return word_set

    def transform(self, corpus=None):

        if corpus is None:
            corpus = self.corpus

        word_freq = np.zeros([corpus.shape[0], self.word_set.shape[0]])

        for i, text in enumerate(corpus):
            text = text.lower().split(' ')
            words, count = np.unique(text, return_counts=True)
            idx = self.word_set.searchsorted(words)

            word_freq[i,idx] = count

        return word_freq

    def fit_transform(self, corpus):

        self.corpus = corpus

        word_list = np.array([])

        word_set = self.fit(self, corpus)
        word_freq = self.transform(self, corpus)

        return word_freq


class VfIdf(object):

    def __init__(self):
        self.N = None
        self.word_set = np.array([])

    def fit(self, corpus):

        self.N = corpus.shape[0]
        self.corpus = corpus
        word_list = np.array([])

        for text in corpus:
            words = text.lower().split(' ')
            word_list = np.append(word_list, words)

        word_set = np.unique(word_list)

        self.word_set = word_set

        return word_set

    def transform(self, corpus=None):

        if corpus is None:
            corpus = self.corpus

        idf = self.IDF(self.word_set, self.corpus)

        tf = self.TF(self.word_set, self.corpus)

        tfidf = idf * tf

        return tfidf

    def fit_transform(self, corpus):
        self.corpus = corpus

        word_set = self.fit(self, corpus)
        tfidf = self.transform(self, corpus)

        return word_set, tfidf


    def TF(self, word_set, corpus):
        word_freq = np.zeros([corpus.shape[0], word_set.shape[0]])

        for i, text in enumerate(corpus):
            text = text.lower().split(' ')
            words, count = np.unique(text, return_counts=True)
            idx = word_set.searchsorted(words)

            word_freq[i, idx] = count
        return word_freq


    def IDF(self, word_set, corpus):

        one_hot_array = np.zeros(shape=[corpus.shape[0], word_set.shape[0]] )

        for i, text in enumerate(corpus):
            words = np.array(text.lower().split(' '))
            one_hot_array[i,:] = np.isin(word_set, words)

        word_rep_docs = np.sum(one_hot_array, axis=0)

        idf = np.log(self.N / word_rep_docs)

        return idf

def cosine_similarity(doc1, doc2):
    
    cs = np.inner(doc1, doc2) / (np.inner(doc1, doc1)**0.5 * np.inner(doc2,doc2)**0.5)
    
    return cs
