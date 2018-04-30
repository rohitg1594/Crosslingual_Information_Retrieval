import numpy as np
import faiss
from collections import defaultdict


def evaluation1(I_test, dico_test):
    '''Evaluation without multiple translations'''

    ks = [1,5,10]
    correct = [0,0,0]
    for i, k in enumerate(ks):
        for j in range(I_test.shape[0]):
            if dico_test[j,1] in I_test[j,:k]:
                correct[i] += 1
        print('Correct : {}, Top {} precision: {}'.format(correct[i], k, correct[i]/I_test.shape[0]))

    return correct


def evaluation2(I_test, dico_test, src_word2id):
    '''Evaluation with multiple translations'''

    ks = [1,5,10]
    dicts = []
    for i, k in enumerate(ks):
        matches = defaultdict(list)
        for j in range(I_test.shape[0]):
            if dico_test[j,1] in I_test[j,:k]:
                matches[src_word2id[dico_test[j,0]]].append(1)
            else:
                matches[src_word2id[dico_test[j,0]]].append(0)

        l = len(matches)
        correct = 0
        for key, value in matches.items():
            matches[key] = max(value)
            correct += max(value)


        dicts.append(dict(matches))
        print('Correct : {}, Top {} precision: {}'.format(correct, k, correct/l))

    return dicts

def eval_data(W, X, tgt_embs, k=20):
    '''Create and search in Faiss Index, return resulting index matrix.'''

    index = faiss.IndexFlatIP(300)
    index.add(tgt_embs.astype(np.float32))
    D, I = index.search((X@W).astype(np.float32), k)

    return I

def evaluation_main(W):
   dico_test = load_dictionary(args.test_dict, -1, src_word2id, tgt_word2id)
   X_test, Y_test = get_parallel_data(src_embs, tgt_embs, dico_test)

   I_test = eval_data(W, X_test, tgt_embs)

   evaluation1(I_test, dico_test)
   dicts = evaluation2(I_test, dico_test, src_id2word)
   incorrect_1 = [k for k, v in dicts[0].items() if v==0]

   for i in range(20):
       src_word = src_id2word[dico_test[i,0]]
       correct_trans = tgt_id2word[dico_test[i,1]]
       if src_word in incorrect_1:
           preds = ''
           for k in range(10):
               pred = tgt_id2word[I_test[i, k]]
               preds += pred + ', '
           preds = preds[:-2]
           print('{:<15}|{:<15}|{}'.format(src_word, correct_trans, preds))

def eval_sents(I, ks):
    '''Evaluate the sentences retrieval for top ks precision.'''
    topks = np.zeros(len(ks))

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if i in I[i,:k]:
                topks[j] += 1

    topks /= I.shape[0]
    return topks
