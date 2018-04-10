from utils import read_emb, load_dictionary, get_parallel_data
from evaluation import evaluation1, evaluation2, eval_data
from procrustes import procrustes
import argparse

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval")
parser.add_argument("--src_embs", default="", help="File path of source embeddings")
parser.add_argument("--tgt_embs", default="", help="File path of target embeddings")
parser.add_argument("--train_dict", default="", help="File path of training dictionary")
parser.add_argument("--test_dict", default="", help="File path of test dictionary")
parser.add_argument("--max_vocab", default=200000, help="Maximum vocabulary size loaded from embeddings")

args = parser.parse_args()

src_embs, src_word2vec, src_word2id, src_id2word = read_emb(args.src_embs, args.max_vocab)
print('Loaded source embeddings')
tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = read_emb(args.tgt_embs, args.max_vocab)
print('Loaded target embeddings')

dico = load_dictionary(args.train_dict, -1, src_word2id, tgt_word2id)
X, Y = get_parallel_data(src_embs, tgt_embs, dico)
W = procrustes(X, Y)

dico_test = load_dictionary(args.test_dict, -1, src_word2id, tgt_word2id)
X_test, Y_test = get_parallel_data(src_embs, tgt_embs, dico_test)
I_test = eval_data(W, X_test, Y_test)
print('Done Searching')


evaluation1(I_test)
evaluation2(I_test, dico_test)
