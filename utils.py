import os
import numpy as np
from sklearn.preprocessing import normalize


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



