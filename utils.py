import os
import numpy as np


def read_emb(path, max_vocab):
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
    print(embs.shape)

    return embs, word2vec, word2id, id2word


def load_dictionary(path, max_vocab, word2id1, word2id2):
    assert os.path.isfile(path)
    dico = np.zeros((max_vocab, 2))
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == max_vocab:
                break
            split = line.split()
            dico[i][0] = int(word2id1[split[0]])
            dico[i][1] = int(word2id2[split[1]])
            

    return dico



