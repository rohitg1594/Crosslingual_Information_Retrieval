import argparse
import os
import pickle
import logging as logging_master


from utils import load_embs, load_embs_bin, load_dictionary, get_parallel_data, str2bool, procrustes
from evaluation import eval_main

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('wikinet')
logging.setLevel(logging_master.INFO)

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval - Learn Word Embedding Mapping",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Word embeddings
parser.add_argument("--src_lang", default="en", help="Source lang")
parser.add_argument("--tgt_lang", default="de", help="Source lang")
parser.add_argument("--save_pickle", default=True, type=str2bool,
                    help="whether to save embeddings as pickle file for faster future recovery")
parser.add_argument("--bin", default=False, type=str2bool, help="whether to load from already saved pickle dump")
parser.add_argument("--emb_dim", default="300", type=int, help="dimension of embeddings")
# Supervised training
parser.add_argument("--data_dir",
                    default="/home/rohit/Documents/Spring_2018/Information_retrieval/Project/"
                            "Crosslingual_Information_Retrieval/data"
                    , help="directory path of data")
parser.add_argument("--max_vocab", default=200000, type=int, help="Maximum vocabulary size loaded from embeddings")
parser.add_argument("--ortho", default=True, type=str2bool,  help="Whether to orthognalize the mapping matrix")
parser.add_argument("--beta", default=0.01, type=float, help="Beta parameter for orthognalization")
parser.add_argument("--norm", default=1, type=int,  help="Normalize embeddings")
# Evaluation
parser.add_argument("--dict", default="MUSE", type=str, choices=["MUSE", "Dinu"], help="set of dictionaries to use")
# Export
parser.add_argument("--export", default=True, type=str2bool, help="whether to export learned mapping matrix")


args = parser.parse_args()

for k, v in vars(args).items():
    print('{:<30}\t{}'.format(k, v))

beta = args.beta
max_vocab = args.max_vocab
norm = args.norm
emb_dim = args.emb_dim
evaluate_every = args.evaluate_every

assert os.path.isdir(args.data_dir)
src_embs_file = os.path.join(args.data_dir, "embs", "wiki." + args.src_lang + ".vec")
tgt_embs_file = os.path.join(args.data_dir, "embs", "wiki." + args.tgt_lang + ".vec")

assert os.path.exists(src_embs_file)
assert os.path.exists(tgt_embs_file)

if args.bin:
    src_path = os.path.join(args.data_dir, "embs", "{}-{}.pickle".format(args.src_lang, args.max_vocab))
    tgt_path = os.path.join(args.data_dir, "embs", "{}-{}.pickle".format(args.tgt_lang, args.max_vocab))
    src_embs, src_word2vec, src_word2id, src_id2word = load_embs_bin(src_path)
    logging.info('Loaded source embeddings')
    tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = load_embs_bin(tgt_path)
    logging.info('Loaded target embeddings')
else:
    src_embs, src_word2vec, src_word2id, src_id2word = load_embs(src_embs_file, max_vocab, norm)
    logging.info('Loaded source embeddings')
    tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = load_embs(tgt_embs_file, max_vocab, norm)
    logging.info('Loaded target embeddings')


# Dictionaries
train_dict = os.path.join(args.data_dir, "dictionaries", "MUSE", args.src_lang + '-' + args.tgt_lang + '.0-5000.txt')
if args.dict == "MUSE":
    test_dict = os.path.join(args.data_dir, "dictionaries", "MUSE",
                             args.src_lang + '-' + args.tgt_lang + '.5000-6500.txt')
else:
    test_dict = os.path.join(args.data_dir, "dictionaries", "Dinu",
                             "OPUS_en_fr_europarl_test.txt",
                             args.src_lang + '-' + args.tgt_lang + '.5000-6500.txt')
assert os.path.exists(train_dict)
assert os.path.exists(test_dict)


if args.save_pickle:
    logging.info("saving loaded embeddings in pickle dump")
    with open(os.path.join(args.data_dir, "embs", '{}-{}.pickle'.format(args.src_lang, args.max_vocab)), 'wb') as f:
        pickle.dump((src_embs, src_word2vec, src_word2id, src_id2word), f)
    with open(os.path.join(args.data_dir, "embs", '{}-{}.pickle'.format(args.tgt_lang, args.max_vocab)), 'wb') as f:
        pickle.dump((tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word), f)
    logging.info("embeddings saved in pickle dump")


dico = load_dictionary(train_dict, -1, src_word2id, tgt_word2id)
X, Y = get_parallel_data(src_embs, tgt_embs, dico)
W = procrustes(X, Y)

if int(args.ortho):
    W = (1 + beta)*W - beta*(W@W.T)@W


logging.info('mapping matrix learned')
logging.info('evaluating final mapping')
eval_main(W, test_dict, src_word2id, tgt_word2id, src_embs, tgt_embs, src_id2word, tgt_id2word, verbose=False)

if args.export:
    f_name = os.path.join(args.data_dir, "mapping", '{}-{}-{}-{}.pickle'.format(args.src_lang, args.tgt_lang,
                                                                                args.max_vocab, args.method))
    logging.info('saving learned mapping to {}'.format(f_name))
    with open(f_name, 'wb') as f:
        pickle.dump(W, f)
