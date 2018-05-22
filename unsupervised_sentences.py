from collections import defaultdict
import pickle

from scipy.linalg import svd
import numpy as np
from sklearn import preprocessing

from utils import sigmoid, tokenize


def load_sentence_bin(path):
    """
    Load sentence corpus from a precomputed pickle dump
    :param path: file path of pickle dump
    :return: list of variable size 1d np.arrays containing token ids in sentence
    """
    with open(path, 'rb') as f:
        corpus = pickle.load(f)

    return corpus


def load_sentence_data(file_name, word2id, max_sents):
    """
    Tokenizes sent using keras, maps the words to id using word2id and returns list of np
    arrays of variable length.
    :param file_name: str :file path of data
    :param word2id: dict : mapping from word to id
    :param chinese: bool : if set to true, uses special chinese tokenizer
    :param export: bool: whether to export sentences as pickle file
    :param bin : bool: load from existing pickle file
    Returns:
    corpus : list of numpy arrays containing word ids of size max_sentences
    """
    corpus = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            if i == max_sents:
                break
            tokens = tokenize(line.strip())
            token_ids = np.array([word2id.get(token.lower(), -1) for token in tokens])
            corpus.append(token_ids)

    return corpus


def mahalanobis(p1, p2, global_cov):
    """Calculate mahalonbis metric b/w p1 and p2"""

    return np.sqrt((p1 - p2) @ global_cov @ (p1 - p2).T)


def cosal_vec(embs, corpus, word2vec, id2word, emb_dim=300, global_only=True, mapper=np.ones((300, 300)), eps=10**-6,
              source=True, norm=True):
    """
    Calculate the CoSal weight sentence embeddings
    Returns:
    numpy ndarray : corpus_vec of same dimension as corpus parameter.
    """
    if source:
        embs = embs@mapper
    global_avg, global_cov = np.mean(embs, axis=0), np.cov(embs, rowvar=False)

    N = len(corpus)
    corpus_vec = np.zeros((N, emb_dim))

    # For every sentence in corpus
    for sent_i, sentence in enumerate(corpus):

        # If sentence is empty, then just put a vector of zeros
        if len(sentence) == 0:
            corpus_vec[sent_i] = np.zeros(emb_dim)
            continue

        # Create array from word ids, if source language, apply mapper
        vecs = np.array([word2vec.get(id2word.get(s, -1), np.zeros(emb_dim)) for s in sentence])
        if source:
            vecs = vecs@mapper

        # Create average vector, use global average if asked
        if global_only:
            avg_vec = global_avg
        else:
            avg_vec = preprocessing.normalize(np.mean(vecs, axis=0)[None], axis=1, norm='l2')

        # Create array of normalized mahalanobis distances
        distances = np.array([mahalanobis(vec, avg_vec, global_cov) for vec in vecs])
        distances /= (2*np.mean(distances, axis=0) + eps)

        # Sigmoid of distances
        if global_only:
            weights = sigmoid(distances)
            weights = weights / np.sum(weights)
        else:
            weights = 1.9*(distances - 0.5) + 0.5

        # Sum out and reshape the output
        vec = np.sum(weights[:, None]*vecs, axis=0)

        # Normalize
        if norm:
            vec = preprocessing.normalize(vec[None], axis=1, norm='l2')

        corpus_vec[sent_i] = vec

    return corpus_vec


def tough_baseline(corpus, word2vec, id2word, word_probs_path, emb_dim=300, a=10**-3, source=True, mapper=np.ones(300),
                   norm=True):
    """
    Compute a simple unsupervised aggregation of word embeddings as described in:
       https://openreview.net/pdf?id=SyK00v5xx
    :param corpus:
    :param word2vec:
    :param id2word:
    :param vec_dim:
    :param a:
    :param source: bool: if set to true apply mapper to word embeddings
    :param mapper:
    :return:
    """
    # Estimate the probabilities of words in the corpus
    with open(word_probs_path, 'rb') as f:
        word_probs = pickle.load(f)

    N = len(corpus)
    corpus_vec = np.zeros((N, emb_dim))

    # Create tf-idf weights
    for sent_idx, sentence in enumerate(corpus):
        vec = np.zeros(emb_dim)

        # For every unique word in sentence
        for word_idx, word_id in enumerate(sentence):
            try:
                if source:
                    vec += a / (a + word_probs.get(word_id, a/10)) * word2vec[id2word[word_id]] @ mapper
                else:
                    vec += a / (a + word_probs.get(word_id, a/10)) * word2vec[id2word[word_id]]
            except KeyError:
                continue

        corpus_vec[sent_idx] = vec

    x = corpus_vec.T
    U, d, v_t = svd(x)
    u = U[:, 0]

    corpus_vec = corpus_vec - corpus_vec@np.outer(u, u.T)

    if norm:
        corpus_vec = preprocessing.normalize(corpus_vec, axis=1, norm='l2')

    return corpus_vec


def tf_idf(corpus, word2vec, id2word, emb_dim=300, mapper=np.ones(300), source=True, norm=True):
    """
    Computes the tf-idf weight for the corpus.
    Returns:
    numpy ndarray : corpus_vec of same dimension as corpus parameter.
    """

    N = len(corpus)
    corpus_vec = np.zeros((N, emb_dim))

    # Estimate the idf in the corpus
    idf_map = defaultdict(int)
    for sent_i, sentence in enumerate(corpus):
        word_indices = np.unique(sentence)
        for word_i in word_indices:
            idf_map[word_i] += 1

    # Create tf-idf weights
    for sent_i, sentence in enumerate(corpus):
        vec = np.zeros(emb_dim)
        index, row_count = np.unique(sentence, return_counts=True)
        index = index.astype(np.int32)
        try:
            f_max = np.max(row_count)
        except ValueError:
            corpus_vec[sent_i] = np.zeros(emb_dim)
            continue

        # For every unique word in sentence
        for i, f in zip(index, row_count):
            idf = np.log(N/idf_map.get(i, 200))

            tf = (1 + np.log(f))/(1 + np.log(f_max))
            try:
                if source:
                    vec += tf*idf*word2vec[id2word[i]]@mapper
                else:
                    vec += tf*idf*word2vec[id2word[i]]
            except KeyError:
                continue

        vec = vec[None]
        if norm:
            vec = preprocessing.normalize(vec, axis=1, norm='l2')
        corpus_vec[sent_i] = vec

    return corpus_vec


def simple_average(corpus, word2vec, id2word, emb_dim=300, mapper=np.ones(300), source=True, norm=True):
    """
    Computes sentence embeddings by taking simple average of word embeddings.
    Returns:
    numpy ndarray : corpus_vec of same dimension as corpus parameter.
    """

    N = len(corpus)
    corpus_vec = np.zeros((N, emb_dim))

    # Create tf-idf weights
    for sent_idx, sentence in enumerate(corpus):
        vec = np.zeros(emb_dim)

        # For every unique word in sentence
        for word_idx, word_id in enumerate(sentence):
            try:
                if source:
                    vec += word2vec[id2word[word_id]] @ mapper
                else:
                    vec += word2vec[id2word[word_id]]
            except KeyError:
                continue

        vec = vec[None]
        if norm:
            vec = preprocessing.normalize(vec, axis=1, norm='l2')
        corpus_vec[sent_idx] = vec

    return corpus_vec
