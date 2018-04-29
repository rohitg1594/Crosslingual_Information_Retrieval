import os
import numpy as np
from sklearn.preprocessing import normalize
from spacy.tokenizer import Tokenizer
import spacy
from collections import defaultdict

np.set_printoptions(edgeitems=5)


def load_embs(path, max_vocab, norm=1):
    '''Read word embeddings from file at path.'''
    assert os.path.isfile(path)
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


def load_dictionary(path, max_vocab, word2id1, word2id2):
    assert os.path.isfile(path)
    dico_list = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == max_vocab:
                break
            split = line.split()
            if split[0] in word2id1 and split[1] in word2id2:
                id1 = int(word2id1[split[0]])
                id2 = int(word2id2[split[1]])
                dico_list.append([id1, id2])

    dico = np.array(dico_list)

    return dico


def get_parallel_data(src_embs, tgt_embs, dico):
    '''Return parallel X and Y matrices corresponding to src
       and tgt embs respectively.'''

    X = src_embs[dico[:, 0].astype(np.int16)]
    Y = tgt_embs[dico[:, 1].astype(np.int16)]

    return X, Y


def load_sentence_data(file_name, model_name, word2id, max_sentences=1000):
    'Read corpus form file and return list of np vectors'
    nlp = spacy.load(model_name)
    tokenizer = Tokenizer(nlp.vocab)

    corpus = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            tokens = tokenizer(line.strip())
            tokens = np.array([word2id.get(token.text.lower(), -1) for token in tokens])
            corpus.append(tokens)
            if i == max_sentences:
                break
    return corpus


def compute_tf_idf(corpus, word2vec, id2word, vec_dim=300, mapper=np.ones(300), source=True, norm=True):
    'Computes the tf-idf weight for the corpus'
    N = len(corpus)
    idfmap = defaultdict(int)
    tf = defaultdict(int)
    corpus_vec = np.zeros((N, vec_dim))

    # Create idfmap
    for sent_i, sentence in enumerate(corpus):
        word_indices = np.unique(sentence)
        for word_i in word_indices:
            idfmap[word_i] += 1

    # Create tf-idf weights
    for sent_i, sentence in enumerate(corpus):
        vec = np.zeros(vec_dim)
        index, row_count = np.unique(sentence, return_counts=True)
        index = index.astype(np.int32)
        try:
            f_max = np.max(row_count)
        except ValueError:
            print('Row count is empty for sentence {}-------->{}'.format(sent_i, sentence))
            corpus_vec[sent_i] = np.zeros(vec_dim)
            continue

        # For every unique word in sentence
        for i, f in zip(index, row_count):
            idf = np.log(N/idfmap[i])
            tf = (1 + np.log(f))/(1 + np.log(f_max))
            try:
                if source:
                    vec += tf*idf*word2vec[id2word[i]]@mapper
                else:
                    vec += tf*idf*word2vec[id2word[i]]
            except KeyError:
                continue

        vec = vec[None]
        if normalize:
            vec = normalize(vec)
        corpus_vec[sent_i] = vec

    return corpus_vec


if __name__ == '__main__':
    import sys

    embs_en, word2vec_en, word2id_en, id2word_en = load_embs('../data/wiki.en.vec', int(sys.argv[2]))
    corpus_en = load_sentence_data(sys.argv[1], 'en', word2id_en)
    compute_tf_idf(corpus_en, word2vec_en, id2word_en)
    sys.exit(0)
    embs_de, word2vec_de, word2id_de, id2word_de = load_embs('../data/wiki.de.vec', int(sys.argv[2]))


