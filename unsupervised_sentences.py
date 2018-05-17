from spacy.lang.zh import Chinese
import spacy
from spacy.tokenizer import Tokenizer

from collections import defaultdict, Counter
import pickle

from scipy.linalg import svd
import numpy as np
from sklearn.preprocessing import normalize

from utils import sigmoid



def load_sentence_bin(path):
    """
    Load sentence corpus from a precomputed pickle dump
    :param path: file path of pickle dump
    :return: list of variable size 1d np.arrays containing token ids in sentence
    """
    with open(path, 'rb') as f:
        corpus = pickle.load(f)

    return corpus


def load_sentence_data(file_name, model_name, word2id, max_sents, chinese=False):
    """
    Tokenizes sent using spacy, maps the words to id using word2id and returns list of np
    arrays of variable length.
    :param file_name: str :file path of data
    :param model_name: str : spacy model name
    :param word2id: dict : mapping from word to id
    :param chinese: bool : if set to true, uses special chinese tokenizer
    :param export: bool: whether to export sentences as pickle file
    :param bin : bool: load from existing pickle file
    Returns:
    corpus : list of numpy arrays containing word ids of size max_sentences
    """
    if chinese:
        nlp = Chinese()
    else:
        nlp = spacy.load(model_name)

    tokenizer = Tokenizer(nlp.vocab)

    corpus = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            tokens = tokenizer(line.strip())
            token_ids = np.array([word2id.get(token.text.lower(), -1) for token in tokens])
            corpus.append(token_ids)
            if i == max_sents:
                break

    return corpus


def mahalanobis(p1, p2, global_cov):
    """Calculate mahalonbis metric b/w p1 and p2"""

    return np.sqrt((p1 - p2)@global_cov@(p1 - p2).T)


def cosal_vec(embs, corpus, word2vec, id2word, emb_dim=300, global_only=True, mapper=np.ones((300, 300)), source=True):
    """
    Calculate the CoSal weight sentence embeddings
    Returns:
    numpy ndarray : corpus_vec of same dimension as corpus parameter.
    """
    if source:
        embs = embs@mapper
    global_avg, global_cov = np.mean(embs, axis=0), np.cov(embs, rowvar=0)

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
            avg_vec = normalize(np.mean(vecs, axis=0)[None], axis=1, norm='l2')

        # Create array of normalized mahalanobis distances
        distances = np.array([mahalanobis(vec, avg_vec, global_cov) for vec in vecs])
        distances /= 2*np.mean(distances)

        # Sigmoid of distances
        if global_only:
            weights = sigmoid(distances)
        else:
            weights = 1.9*(distances - 0.5) + 0.5

        print(weights)
        # Sum out and reshape the output
        corpus_vec[sent_i] = np.sum(weights.reshape(weights.shape[0], -1)*vecs, axis=0)

    return corpus_vec


def _word_probs(corpus):
    """
    Returns the estimated word probabilites of the corpus
    :param corpus: list of sentences in the corpus, words in the form of ids
    :return: c : dic of the form => word :  probability
    """
    c = Counter()
    corpus = [word for sent in corpus for word in sent]
    total_words = len(corpus)
    c = Counter(corpus)

    c = {word: count / total_words for word, count in c.items()}

    return dict(c)


def tough_baseline(corpus, word2vec, id2word, word_probs_path, emb_dim, a=10**-3, source=True, mapper=np.ones(300)):
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
    assert emb_dim == len(mapper)
    corpus_vec = np.zeros((len(corpus), emb_dim))

    # Create map id2vec
    id2vec = {}
    for id, word in id2word.items():
        if word in word2vec:
            id2vec[id] = word2vec[word]

    # Estimate the probabilities of words in the corpus
    #word_probs = _word_probs(corpus)
    with open(word_probs_path, 'rb') as f:
        word_probs = pickle.load(f)

    for idx_sent, sent in enumerate(corpus):
        if len(sent) == 0:
            corpus_vec[idx_sent] = np.zeros(emb_dim)
        else:
            corpus_vec[idx_sent] = 1 / len(sent) * sum([a / (a + word_probs[word_id]) * id2vec[word_id]
                                                        for word_id in sent if word_id != -1 and word_id in word_probs
                                                        and word_id in id2vec])

    x = corpus_vec.T
    U, d, v_t = svd(x)
    u = U[:, 0]

    for idx_sent, sent in enumerate(corpus_vec):
        corpus_vec[idx_sent] = sent - np.outer(u, u.T)@sent

    if source:
        corpus_vec = corpus_vec@mapper

    return corpus_vec


def compute_tf_idf(corpus, word2vec, id2word, vec_dim=300, mapper=np.ones(300), source=True, norm=True):
    """
    Computes the tf-idf weight for the corpus.
    Returns:
    numpy ndarray : corpus_vec of same dimension as corpus parameter.
    """

    N = len(corpus)
    idf_map = defaultdict(int)
    corpus_vec = np.zeros((N, vec_dim))

    # Create idfmap
    for sent_i, sentence in enumerate(corpus):
        word_indices = np.unique(sentence)
        for word_i in word_indices:
            idf_map[word_i] += 1

    # Create tf-idf weights
    for sent_i, sentence in enumerate(corpus):
        vec = np.zeros(vec_dim)
        index, row_count = np.unique(sentence, return_counts=True)
        index = index.astype(np.int32)
        try:
            f_max = np.max(row_count)
        except ValueError:
            corpus_vec[sent_i] = np.zeros(vec_dim)
            continue

        # For every unique word in sentence
        for i, f in zip(index, row_count):
            idf = np.log(N/idf_map[i])
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
            vec = normalize(vec, axis=1, norm='l2')
        corpus_vec[sent_i] = vec

    return corpus_vec
