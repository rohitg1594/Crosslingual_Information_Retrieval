import subprocess
import os
from os.path import join
import logging as logging_master
import argparse

from utils import load_embs, calc_word_probs
from aggregation import load_sentence_data
import pickle

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_dir", default="data", help="directory path of data")
parser.add_argument("--exp_name", default="", help="experiment name, results of experiment will be saved in this file")
parser.add_argument("--env_path", default="/home/rohit/anaconda3/envs/InfoRetrieval36", type=str,
                    help="directory path of virtual or conda environment")
args = parser.parse_args()

DATA_PATH = args.data_dir

my_env = os.environ.copy()
my_env["PATH"] = args.env_path
my_cwd = os.path.dirname(os.path.realpath(__file__))

langs = ['en', 'es', 'it', 'de', 'fr', 'fi']
embs_path = {}
embs_bin_path = {}
sent_path = {}

f = open(join(DATA_PATH, "experiments", "words-{}.txt".format(args.exp_name)), 'w')
f.close()

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)


for lang in langs:
    if lang == 'en':
        continue
    embs_path[lang] = join(DATA_PATH, 'embs', 'wiki.{}.vec'.format(lang))
    embs_bin_path[lang] = join(DATA_PATH, 'embs', '{}-200000.pickle'.format(lang))
    sent_path[lang] = join(DATA_PATH, 'europarl', 'europarl-v7.{}-en.{}'.format(lang, lang))

for k, v in embs_path.items():
    embs, word2vec, word2id, id2word = load_embs(v, 200000)
    logging.info('Loaded word embeddings for {}'.format(k))

    corpus = load_sentence_data(sent_path[k], word2id, 200000)
    logging.info('Loaded sentence corpora for {}'.format(k))

    word_probs = calc_word_probs(corpus)
    logging.info('Calculated word probabilities for {}'.format(k))

    with open(join(DATA_PATH, "europarl", "word-probs-{}.pickle".format(k)), 'wb') as f:
        pickle.dump(word_probs, f)
    logging.info('Saved word probabilities for {}'.format(k))


def experiment(src_lang, tgt_lang):
    call_str = args.env_path + "bin/python "
    call_str += "words.py --src_lang {} --tgt_lang {}".format(src_lang, tgt_lang)
    out = subprocess.check_output(call_str.split(), stderr=subprocess.STDOUT, env=my_env, cwd=my_cwd)
    try:
        with open(join(DATA_PATH, "experiments", "words-{}.txt".format(args.exp_name)), 'a') as f_out:
            f_out.write('{}-{}\n'.format(src_lang, tgt_lang))
            f_out.write(out.decode('ascii'))
            f_out.write('\n')
    except subprocess.CalledProcessError as e:
        print(e.output.decode('ascii'))


for lang in langs:
    if lang == 'en':
        continue
    logging.info('Starting experiment {}, {}'.format(lang, 'en'))
    experiment(lang, 'en')
    logging.info('Done with experiment.')

    logging.info('Starting experiment {}, {}'.format('en', lang))
    experiment('en', lang)
    logging.info('Done with experiment.')
