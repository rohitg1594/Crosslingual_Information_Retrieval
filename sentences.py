import argparse
import faiss
import numpy as np
import spacy
import os
from os.path import join
import sys
import logging as logging_master
import pickle

from utils import load_embs, load_embs_bin, str2bool
from unsupervised_sentences import compute_tf_idf, load_sentence_data, cosal_vec, tough_baseline, load_sentence_bin
from evaluation import eval_sents


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
parser.add_argument("--src_sents", default="data/europarl-v7.de-en.en", help="file path of source sentences")
parser.add_argument("--tgt_sents", default="data/europarl-v7.de-en.de", help="file path of target sentences")
parser.add_argument("--sent_bin", default=True, type=str2bool, help="whether to load sentence embeddings from pickle dump")
parser.add_argument("--max_sents", default=1000, type=int, help="maximum number of sentences loaded from disk")
parser.add_argument("--aggr_sents", default='tf-idf', choices=['CoSal', 'tf-idf', 'tough_baseline'],
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
assert os.path.exists(args.map_file)

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

with open(args.map_file, 'rb') as f:
    W = pickle.load(f)
    logging.info("Loaded mapper")

try:
    if args.src_lang != 'zh':
        src_nlp = spacy.load(args.src_lang)
except IOError:
    logging.error("You don't have {} spacy model on your machine".format(args.src_lang))
    logging.error("Please download from the command line: python -m spacy download {}".format(args.src_lang))
    logging.error("Exiting now!!!!")
    sys.exit(1)

try:
    if args.tgt_lang != 'zh':
        tgt_nlp = spacy.load(args.tgt_lang)
except IOError:
    logging.error("You don't have {} spacy model on your machine".format(args.tgt_lang))
    logging.error("Please download from the command line: python -m spacy download {}".format(args.tgt_lang))
    logging.error("Exiting now!!!!")
    sys.exit(1)

if args.src_lang == 'zh' or args.tgt_lang == 'zh':
    try:
        from spacy.lang.zh import Chinese
    except IOError:
        logging.error("You don't have jieba on your machine")
        logging.error("Please download from the command line: pip install jieba")
        logging.error("Exiting now!!!!")
        sys.exit(1)

if args.src_lang == 'zh':
    src_chinese = True
else:
    src_chinese = False

if args.tgt_lang == 'zh':
    tgt_chinese = True
else:
    tgt_chinese = False

if args.sent_bin:
    src_path = os.path.join(args.data_dir, "europarl", "{}-{}.pickle".format(args.src_lang, args.max_sents))
    tgt_path = os.path.join(args.data_dir, "europarl", "{}-{}.pickle".format(args.tgt_lang, args.max_sents))
    src_corpus = load_sentence_bin(src_path)
    logging.info('Loaded source corpus')
    tgt_corpus = load_sentence_bin(tgt_path)
    logging.info('Loaded target corpus')
else:
    src_corpus = load_sentence_data(args.src_sents, args.src_lang, src_word2id, max_sents=args.max_sents, chinese=src_chinese)
    logging.info('Loaded source corpus')
    tgt_corpus = load_sentence_data(args.tgt_sents, args.tgt_lang, tgt_word2id, max_sents=args.max_sents, chinese=tgt_chinese)
    logging.info('Loaded target corpus')


if args.export:
    f_name = os.path.join(args.data_dir, "europarl", '{}-{}.pickle'.format(args.src_lang, args.max_sents))
    logging.info('saving corpus to {}'.format(f_name))
    with open(f_name, 'wb') as f:
        pickle.dump(src_corpus, f)
    f_name = os.path.join(args.data_dir, "europarl", '{}-{}.pickle'.format(args.tgt_lang, args.max_sents))
    logging.info('saving corpus to {}'.format(f_name))
    with open(f_name, 'wb') as f:
        pickle.dump(tgt_corpus, f)

if args.aggr_sents == 'tf-idf':
    src_vec = compute_tf_idf(src_corpus, src_word2vec, src_id2word, mapper=W, source=True)
    tgt_vec = compute_tf_idf(tgt_corpus, tgt_word2vec, tgt_id2word, source=False)

elif args.aggr_sents == 'CoSal':
    src_vec = cosal_vec(src_embs, src_corpus, src_word2vec, src_id2word, mapper=W, emb_dim=args.vec_dim, global_only=False, source=True)
    tgt_vec = cosal_vec(tgt_embs, tgt_corpus, tgt_word2vec, tgt_id2word, emb_dim=args.vec_dim, global_only=False, source=False)

elif args.aggr_sents == 'tough_baseline':
    src_vec = tough_baseline(src_corpus, src_word2vec, src_id2word,
                             word_probs_path=join(args.data_dir, "europarl", "word-probs-{}.pickle".format(args.src_lang)),
                             emb_dim=args.emb_dim, mapper=W, source=True)
    tgt_vec = tough_baseline(tgt_corpus, tgt_word2vec, tgt_id2word,
                             word_probs_path=join(args.data_dir, "europarl", "word-probs-{}.pickle".format(args.src_lang)),
                             emb_dim=args.emb_dim, source=False)

index = faiss.IndexFlatIP(args.emb_dim)
index.add(tgt_vec.astype(np.float32))
D, I = index.search(src_vec.astype(np.float32), 10)

eval_sents(I, [1, 5, 10])
