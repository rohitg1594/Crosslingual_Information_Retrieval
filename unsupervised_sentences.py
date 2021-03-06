import argparse
import faiss
import numpy as np
import os
from os.path import join
import sys
import logging as logging_master
import pickle

from utils import load_embs, load_embs_bin, str2bool
from aggregation import tf_idf, load_sentence_data, cosal_vec, tough_baseline, simple_average, max_pool
from evaluate_unsupervised import eval_sents


logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('wikinet')
logging.setLevel(logging_master.INFO)

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Word embeddings
parser.add_argument("--src_lang", default="en", help="Source lang")
parser.add_argument("--tgt_lang", default="de", help="Source lang")
parser.add_argument("--word_bin", default=True, type=str2bool, help="whether to load word embeddings from pickle dump")
parser.add_argument("--emb_dim", default="300", type=int, help="dimension of embeddings")
parser.add_argument("--data_dir",
                    default="/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data"
                    , help="directory path of data")
parser.add_argument("--map_file",default="", help="file path of map from scr to tgt lang")
parser.add_argument("--max_vocab", default=200000, type=int, help="Maximum vocabulary size loaded from embeddings")
parser.add_argument("--norm", default=1, type=int,  help="Normalize embeddings")
# Sentence Retrieval
parser.add_argument("--max_sents", default=10000, type=int, help="maximum number of sentences loaded from disk")
parser.add_argument("--method", default='tf-idf', choices=['tf-idf',  'simple_average', 'CoSal', 'tough_baseline',
                                                           'max_pool'],
                    help="type of sentence aggregation method")
# Evaluation
parser.add_argument("--eval_sents", default=10000, type=int, help="how many sentences to evaluate on")


args = parser.parse_args()

for k, v in vars(args).items():
    print('{:<30}\t{}'.format(k, v))

src_embs_file = os.path.join(args.data_dir, "embs", "wiki." + args.src_lang + ".vec")
tgt_embs_file = os.path.join(args.data_dir, "embs", "wiki." + args.tgt_lang + ".vec")

assert os.path.isdir(args.data_dir)
assert os.path.exists(src_embs_file)
assert os.path.exists(tgt_embs_file)

# Word Embeddings
if args.word_bin:
    src_path = os.path.join(args.data_dir, "embs", "{}-{}.pickle".format(args.src_lang, args.max_vocab))
    tgt_path = os.path.join(args.data_dir, "embs", "{}-{}.pickle".format(args.tgt_lang, args.max_vocab))
    src_embs, src_word2vec, src_word2id, src_id2word = load_embs_bin(src_path)
    logging.info('Loaded source embeddings')
    tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = load_embs_bin(tgt_path)
    logging.info('Loaded target embeddings')
else:
    src_embs, src_word2vec, src_word2id, src_id2word = load_embs(src_embs_file, args.max_vocab, args.norm)
    logging.info('Loaded source embeddings')
    tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = load_embs(tgt_embs_file, args.max_vocab, args.norm)
    logging.info('Loaded target embeddings')

# Mapper
map_file = join(args.data_dir, "mapping", "{}-{}-200000.pickle".format(args.src_lang, args.tgt_lang))
with open(map_file, 'rb') as f:
    W = pickle.load(f)
    logging.info("Loaded mapper")


src_path = os.path.join(args.data_dir, "europarl",
                        "europarl-v7.{}-{}.{}".format(args.src_lang, args.tgt_lang, args.src_lang))
if not os.path.exists(src_path):
    src_path = os.path.join(args.data_dir, "europarl",
                            "europarl-v7.{}-{}.{}".format(args.tgt_lang, args.src_lang, args.src_lang))
src_corpus = load_sentence_data(src_path, src_word2id, max_sents=args.max_sents)
logging.info('Loaded source corpus')

tgt_path = os.path.join(args.data_dir, "europarl",
                        "europarl-v7.{}-{}.{}".format(args.src_lang, args.tgt_lang, args.tgt_lang))
if not os.path.exists(tgt_path):
    tgt_path = os.path.join(args.data_dir, "europarl",
                            "europarl-v7.{}-{}.{}".format(args.tgt_lang, args.src_lang, args.tgt_lang))
tgt_corpus = load_sentence_data(tgt_path, tgt_word2id, max_sents=args.max_sents)
logging.info('Loaded target corpus')


# Aggregation
if args.method == 'tf-idf':
    src_vec = tf_idf(src_corpus, src_word2vec, src_id2word, mapper=W, source=True)
    tgt_vec = tf_idf(tgt_corpus, tgt_word2vec, tgt_id2word, source=False)

elif args.method == 'simple_average':
    src_vec = simple_average(src_corpus, src_word2vec, src_id2word, mapper=W, source=True)
    tgt_vec = simple_average(tgt_corpus, tgt_word2vec, tgt_id2word, source=False)

elif args.method == 'CoSal':
    src_vec = cosal_vec(src_embs, src_corpus, src_word2vec,
                        src_id2word, mapper=W, emb_dim=args.emb_dim, global_only=True, source=True)
    tgt_vec = cosal_vec(tgt_embs, tgt_corpus, tgt_word2vec,
                        tgt_id2word, emb_dim=args.emb_dim, global_only=True, source=False)

elif args.method == 'tough_baseline':
    src_vec = tough_baseline(src_corpus, src_word2vec, src_id2word,
                             word_probs_path=join(args.data_dir, "europarl", "word-probs-{}.pickle".format(args.src_lang)),
                             emb_dim=args.emb_dim, mapper=W, source=True)
    tgt_vec = tough_baseline(tgt_corpus, tgt_word2vec, tgt_id2word,
                             word_probs_path=join(args.data_dir, "europarl", "word-probs-{}.pickle".format(args.src_lang)),
                             emb_dim=args.emb_dim, source=False)

elif args.method == 'max_pool':
    src_vec = max_pool(src_corpus, src_word2vec, src_id2word, mapper=W, source=True)
    tgt_vec = max_pool(tgt_corpus, tgt_word2vec, tgt_id2word, source=False)
else:
    logging.error("aggregation method {} not supported!".format(args.aggr_sents))
    sys.exit(1)

# Evaluation
index = faiss.IndexFlatIP(args.emb_dim)
index.add(tgt_vec.astype(np.float32))
D, I = index.search(src_vec.astype(np.float32), 10000)

for i in range(20):
    true = src_corpus[i]
    true_str = ' '.join([src_id2word[id] for id in true])
    print(true_str)
    print('---------------------')
    pred_str = ''
    predictions = I[i][:10]
    for prediction in predictions:
        tgt_sent = tgt_corpus[prediction]
        pred_str = ' '.join([tgt_id2word[id] for id in tgt_sent])
        print(pred_str)
    print('\n\n\n')


eval_sents(I, [1, 5, 10])
