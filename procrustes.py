import numpy as np
from scipy.linalg import svd
from scipy.linalg import norm
from utils import read_emb, load_dictionary
from sklearn.preprocessing import normalize
import faiss

def procrustes_training(max_vocab):
    src_embs, src_word2vec, src_word2id, src_id2word = read_emb('../data/wiki.en.vec', max_vocab)
    print('Loaded source embeddings')
    tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = read_emb('../data/wiki.de.vec', max_vocab)
    print('Loaded target embeddings')

    dico = load_dictionary('../MUSE/data/crosslingual/dictionaries/en-de.txt', 50000, src_word2id, tgt_word2id)

    X = src_embs[dico[:, 0].astype(np.int16)]
    Y = tgt_embs[dico[:, 1].astype(np.int16)]

    X = normalize(X, axis=1, norm='l2')
    Y = normalize(Y, axis=1, norm='l2')

    print(X.shape)
    print(Y.shape)

    M = X.T@Y
    U, S, V_T = svd(M, full_matrices=True )
    W = U.dot(V_T)


    ## Evaluation
    dico_test = load_dictionary('../MUSE/data/crosslingual/dictionaries/en-de.5000-6500.txt', 1500, src_word2id, tgt_word2id)
    X_test = src_embs[dico_test[:, 0].astype(np.int16)]
    Y_test = tgt_embs[dico_test[:, 1].astype(np.int16)]

    X_test = normalize(X_test, axis=1, norm='l2')
    Y_test = normalize(Y_test, axis=1, norm='l2')
    index = faiss.IndexFlatIP(300)
    index.add(Y_test.astype(np.float32))

    D_test, I_test = index.search((X_test@W).astype(np.float32), 20)
    print('Done Searching')
    print(I_test[:20])

    ks = [1,5,10]
    correct = [0,0,0]
    for i, k in enumerate(ks):
        for j in range(I_test.shape[0]):
            if j in I_test[j,:k]:
                correct[i] += 1
    print(correct)
    print(np.array(correct)/len(I_test))


if __name__ == "__main__":
    procrustes_training(200000)
