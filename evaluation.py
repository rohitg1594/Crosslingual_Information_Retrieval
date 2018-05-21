import faiss
from utils import *
from collections import defaultdict


def eval_wo(I_test, dico_test):
    """Evaluation without multiple translations"""

    ks = [1,5,10]
    correct = [0,0,0]
    for i, k in enumerate(ks):
        for j in range(I_test.shape[0]):
            if dico_test[j,1] in I_test[j,:k]:
                correct[i] += 1
        print('Correct : {}, Top {} precision: {:.3f}'.format(correct[i], k, correct[i]/I_test.shape[0]))

    return correct


def eval_w(I_test, dico_test, src_word2id):
    """Evaluation with multiple translations"""

    ks = [1, 5, 10]
    dicts = []
    for i, k in enumerate(ks):
        matches = defaultdict(list)
        for j in range(I_test.shape[0]):
            if dico_test[j, 1] in I_test[j, :k]:
                matches[src_word2id[dico_test[j, 0]]].append(1)
            else:
                matches[src_word2id[dico_test[j, 0]]].append(0)

        num_matches = len(matches)
        correct = 0
        for key, value in matches.items():
            matches[key] = max(value)
            correct += max(value)


        dicts.append(dict(matches))
        print('Correct : {}, Top {} precision: {:.3f}'.format(correct, k, correct/num_matches))

    return dicts


def eval_data(w, x, tgt_embs, k=20):
    """Create and search in Faiss Index, return resulting index matrix."""

    index = faiss.IndexFlatIP(300)
    index.add(tgt_embs.astype(np.float32))
    d, i = index.search((x@w).astype(np.float32), k)

    return i


def eval_main(W, test_dict, src_word2id, tgt_word2id, src_embs, tgt_embs, src_id2word, tgt_id2word, verbose=False):
    """Main evaluation function for evaluating word embedding, uses both eval_w and eval_wo"""

    dico_test = load_dictionary(test_dict, -1, src_word2id, tgt_word2id)
    X_test, Y_test = get_parallel_data(src_embs, tgt_embs, dico_test)

    I_test = eval_data(W, X_test, tgt_embs)

    print("Taking into account multiple translations:")
    eval_wo(I_test, dico_test)
    print("Not taking into account multiple translations:")
    dicts = eval_w(I_test, dico_test, src_id2word)
    incorrect_1 = [k for k, v in dicts[0].items() if v == 0]

    # print out incorrect translations for analysis and debugging
    if verbose:
        for i in range(20):
           src_word = src_id2word[dico_test[i,0]]
           correct_trans = tgt_id2word[dico_test[i,1]]
           if src_word in incorrect_1:
               preds = ''
               for k in range(10):
                   pred = tgt_id2word[I_test[i, k]]
                   preds += pred + ', '
               preds = preds[:-2]
               print('{:<15}|{:<15}|{:.3f}'.format(src_word, correct_trans, preds))


def eval_sents(I, ks):
    """Evaluate the sentences retrieval for top ks precision."""
    topks = np.zeros(len(ks))

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if i in I[i,:k]:
                topks[j] += 1

    topks /= I.shape[0]

    for k, topk in zip(ks, topks):
        print('Top {} precision : {:.3f}'.format(k, topk))

    # potential_ks = 5*np.arange(3, 1000)
    # eps = 10**-3
    # matches = np.zeros(len(potential_ks))
    # for j, k in enumerate(potential_ks):
    #     for i in range(I.shape[0]):
    #         if i in I[i,:k]:
    #             matches[j] += 1
    #     precision = matches[j]/I.shape[0]
    #     print('Precision at {} is {}'.format(k, precision))
    #     if 1 - precision < eps:
    #         break
    # print('Full matches at k = {}'.format(k))