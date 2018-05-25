import os
import pickle
import argparse
from collections import Counter
import logging as logging_master

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import svd
from sklearn.cross_decomposition import CCA

from keras.preprocessing.text import text_to_word_sequence

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('wikinet')
logging.setLevel(logging_master.INFO)

np.set_printoptions(edgeitems=5)

MAX_SENT_SIZE = 50


def tokenize(text):
    sent = text_to_word_sequence(text, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    return sent


def calc_word_probs(corpus):
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


def load_embs_bin(path):
    """Read word embeddings from pickle file at path.
    :param: path : str :file path of word embedding in pickle format

    """
    with open(path, 'rb') as f:
        embs, word2vec, word2id, id2word = pickle.load(f)

    return embs, word2vec, word2id, id2word


def load_embs(path, max_vocab, norm=True):
    """Read word embeddings from vec file at path.
    :param: path : str :file path of word embedding in txt format
    :param: max_vocab : int : max number of word embeddings loaded from disk
    :param: norm : bool : whether to normalize matrices
    """
    word2vec = {}
    word2id = {}
    vectors = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):

            if len(word2vec) >= max_vocab:
                break
            if i == 0:
                vocab_size, emb_size = line.split()
            else:
                word, vec = line.split(' ', 1)
                vec = np.fromstring(vec.strip(), sep=' ')
                assert len(vec) == int(emb_size)
                if word in word2vec:
                    print("Same word encountered twice\n")
                    continue
                else:
                    word2vec[word] = vec
                    word2id[word] = int(len(word2id))
                    vectors.append(vec[None])

    id2word = {v:k for k,v in word2id.items()}
    embs = np.concatenate(vectors, 0)

    if norm:
        embs = normalize(embs, axis=1, norm='l2')

    return embs, word2vec, word2id, id2word


def load_dictionary(path, max_vocab, src_word2id, tgt_word2id):
    """Loads training dict from file at path."""

    assert os.path.isfile(path)
    dico_list = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == max_vocab:
                break
            split = line.split()
            if split[0] in src_word2id and split[1] in tgt_word2id:
                id1 = int(src_word2id[split[0]])
                id2 = int(tgt_word2id[split[1]])
                dico_list.append([id1, id2])

    dico = np.array(dico_list)

    return dico


def get_parallel_data(src_embs, tgt_embs, dico):
    """Return parallel X and Y matrices corresponding to source
       and target embedding matrices respectively."""

    X = src_embs[dico[:, 0].astype(np.int16)]
    Y = tgt_embs[dico[:, 1].astype(np.int16)]

    return X, Y


def sigmoid(x):
    """Sigmoid function"""

    return 1/(1 + np.exp(-x))


def str2bool(v):
    """
    thanks : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def procrustes(x, y):
    """
    Solve for linear mapping using procrustes solution.
    """
    m = x.T@y
    u, s, v_T = svd(m, full_matrices=True )
    w = u.dot(v_T)

    return w


def cca(x, y, num_components):
    """
    Transform x and y using sklearn's cross correlation analysis.
    :param x: source language embeddings
    :param y: target language embeddings
    :param num_components: number of components to keep
    :return: x_c and y_c transformed matrices
    """
    cca = CCA(n_components=num_components)
    logging.info("Learning CCA decomposition")
    cca.fit(x, y)
    logging.info("CCA decomposion learned")

    return cca.transform(x, y)


def _pad_sent(sent):
    if len(sent) < MAX_SENT_SIZE:
        sent = np.pad(sent, (0, MAX_SENT_SIZE - len(sent)), 'constant')
    else:
        sent = sent[:MAX_SENT_SIZE]
    return sent


def create_padded_data(corpus):
    out = np.zeros((len(corpus), MAX_SENT_SIZE))
    for idx_sent, sent in enumerate(corpus):
        out[idx_sent] = _pad_sent(sent)

    return out
