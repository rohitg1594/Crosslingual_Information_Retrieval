from utils import read_emb, load_dictionary, get_parallel_data
from evaluation import evaluation1, evaluation2, eval_data
from procrustes import procrustes
import argparse
import faiss
import numpy as np

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval")
parser.add_argument("--src_embs", default="", help="File path of source embeddings")
parser.add_argument("--tgt_embs", default="", help="File path of target embeddings")
parser.add_argument("--train_dict", default="", help="File path of training dictionary")
parser.add_argument("--test_dict", default="", help="File path of test dictionary")
parser.add_argument("--max_vocab", default=200000, help="Maximum vocabulary size loaded from embeddings")
parser.add_argument("--orthognalize", default=0, help="Whether to orthognolize the mapping matrix")
parser.add_argument("--beta", default=0.01, help="Beta parameter for orthognalization")

args = parser.parse_args()
beta = float(args.beta)

assert int(args.orthognalize) in [0, 1]

src_embs, src_word2vec, src_word2id, src_id2word = read_emb(args.src_embs, args.max_vocab)
print('Loaded source embeddings')
tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = read_emb(args.tgt_embs, args.max_vocab)
print('Loaded target embeddings')

dico = load_dictionary(args.train_dict, -1, src_word2id, tgt_word2id)
X, Y = get_parallel_data(src_embs, tgt_embs, dico)
W = procrustes(X, Y)

if int(args.orthognalize):
    W = (1 + beta)*W - beta*(W@W.T)@W

dico_test = load_dictionary(args.test_dict, -1, src_word2id, tgt_word2id)
X_test, Y_test = get_parallel_data(src_embs, tgt_embs, dico_test)

I_test = eval_data(W, X_test, tgt_embs)

evaluation1(I_test, dico_test)
dicts = evaluation2(I_test, dico_test, src_id2word)
incorrect_1 = [k for k, v in dicts[0].items() if v==0]

for i in range(1000):
    src_word = src_id2word[dico_test[i,0]]
    correct_trans = tgt_id2word[dico_test[i,1]]
    if src_word in incorrect_1:
        preds = ''
        for k in range(5):
            pred = tgt_id2word[I_test[i, k]]
            preds += pred + ', '
        preds = preds[:-2]
        print('{:<15}|{:<15}|{}'.format(src_word, correct_trans, preds))

print('Done Searching')



