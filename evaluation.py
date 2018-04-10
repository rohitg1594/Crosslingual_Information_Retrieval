import numpy as np
import faiss
from collections import defaultdict


def evaluation1(I_test):
    '''Evaluation without multiple translations'''

    ks = [1,5,10]
    correct = [0,0,0]
    for i, k in enumerate(ks):
        for j in range(I_test.shape[0]):
            if j in I_test[j,:k]:
                correct[i] += 1
        print('Correct : {}, Top {} precision: {}'.format(correct[i], k, correct[i]/I_test.shape[0]))

    return correct


def evaluation2(I_test, dico_test):
    '''Evaluation with multiple translations'''

    ks = [1,5,10]
    for i, k in enumerate(ks):
        matches = defaultdict(list)
        for j in range(I_test.shape[0]):
            if j in I_test[j,:k]:
                matches[dico_test[j,0]].append(1)
            else:
                matches[dico_test[j,0]].append(0)

        l = len(matches)
        correct = 0
        for value in matches.values():
            correct += max(value)
        print('Correct : {}, Top {} precision: {}'.format(correct, k, correct/l))
    
    
def eval_data(W, X, Y, k=20):
    '''Create and search in Faiss Index, return resulting index matrix.'''

    index = faiss.IndexFlatIP(300)
    index.add(Y.astype(np.float32))
    D, I = index.search((X@W).astype(np.float32), k)

    return I
    
