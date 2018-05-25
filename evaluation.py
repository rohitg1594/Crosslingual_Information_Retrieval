import faiss
from utils import load_dictionary, get_parallel_data
import numpy as np
from collections import defaultdict


def eval_wo(pred, dico):
    """Evaluation without multiple translations.
    :param: pred: np.ndarray : predictions where each row contains indices of predictions
    :param: dico_test: np.ndarray : ndarray with same number of rows as I_test and two columns. Second column
                                    contains true values.
    :return: correct : list :  containing total number of correct predictions in top 1, 5, and 10.
    """

    ks = [1, 5, 10]
    correct = [0, 0, 0]
    for i, k in enumerate(ks):
        for j in range(pred.shape[0]):
            if dico[j, 1] in pred[j, :k]:
                correct[i] += 1
        print('Correct : {}, Top {} precision: {:.3f}'.format(correct[i], k, correct[i]/pred.shape[0]))

    return correct


def eval_w(pred, dico, src_id2word):
    """
    Evaluation with multiple translations
    :param: I: np.ndarray : predictions where each row contains indices of predictions
    :param: dico: np.ndarray : ndarray with same number of rows as I_test and two columns. Second column
                                    contains true values.
    :param: src_id2word : dict : map from integer id to word.
    :return: dicts : list of dicts :  each dict has as key the wordid for a match in pred
    """

    ks = [1, 5, 10]
    dicts = []
    for i, k in enumerate(ks):
        matches = defaultdict(list)
        for j in range(pred.shape[0]):
            if dico[j, 1] in pred[j, :k]:
                matches[src_id2word[dico[j, 0]]].append(1)
            else:
                matches[src_id2word[dico[j, 0]]].append(0)

        num_matches = len(matches)
        correct = 0
        for key, value in matches.items():
            matches[key] = max(value)
            correct += max(value)

        dicts.append(dict(matches))
        print('Correct : {}, Top {} precision: {:.3f}'.format(correct, k, correct/num_matches))

    return dicts


def _eval_data(w, x, y, k=20):
    """
    Helper function to create and search in Faiss Index, return resulting index matrix.
    :param: w : np.ndarray : mapper from src to tgt space
    :param: x : np.ndarray : embeddings in src space
    :param: y : np.ndarray : embeddings in tgt space
    :param: k : number of closest neighbours returned by Faiss
    :return : i : np.ndarray : index matrix containing predictions
    """

    index = faiss.IndexFlatIP(300)
    index.add(y.astype(np.float32))
    d, i = index.search((x@w).astype(np.float32), k)

    return i


def eval_main(W, test_dict, src_word2id, tgt_word2id, src_embs, tgt_embs, src_id2word, tgt_id2word, verbose=False):
    """Main evaluation function for evaluating word embedding, uses both eval_w and eval_wo.
    :param: W : np.ndarray :  mapping matrix of shape emb_dim \times emb_dim
    :param: test
    """

    dico_test = load_dictionary(test_dict, -1, src_word2id, tgt_word2id)
    x_test, y_test = get_parallel_data(src_embs, tgt_embs, dico_test)

    i_test = _eval_data(W, x_test, tgt_embs)

    print("Taking into account multiple translations:")
    eval_wo(i_test, dico_test)
    print("Not taking into account multiple translations:")
    dicts = eval_w(i_test, dico_test, src_id2word)
    incorrect_1 = [k for k, v in dicts[0].items() if v == 0]

    # print out incorrect translations for analysis and debugging
    if verbose:
        for i in range(20):
            src_word = src_id2word[dico_test[i,0]]
            correct_trans = tgt_id2word[dico_test[i,1]]
            if src_word in incorrect_1:
                preds = ''
                for k in range(10):
                    pred = tgt_id2word[i_test[i, k]]
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
        print('Top {} precision : {:.7f}'.format(k, topk))

    # Mean Reciprocal Rank
    ranks = []
    for i in range(I.shape[0]):
        ranks.append(1 / (np.where(i == I[i])[0] + 1))
    print('Mean Reciprocal Rank : {}'.format(np.mean(np.array(ranks))))