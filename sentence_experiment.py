import subprocess
import os
from os.path import join
import logging as logging_master
import pickle
import argparse

from utils import load_embs
from aggregation import load_sentence_data
from utils import calc_word_probs

my_env = os.environ.copy()
my_cwd = os.path.dirname(os.path.realpath(__file__))

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)


parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_dir", default="data", help="directory path of data")
parser.add_argument("--exp_name", default="", help="experiment name, results of experiment will be saved in this file")
parser.add_argument("--max_sent", default=300000, type=int, help="number of sentences to estimate statistics")
parser.add_argument("--eval_sent", default=10000, type=int, help="number of sentences to evaluate on")
parser.add_argument("--env_path", default="/home/rohit/anaconda3/envs/InfoRetrieval36", type=str,
                    help="directory path of virtual or conda environment")
args = parser.parse_args()

my_env["PATH"] = args.env_path
DATA_PATH = args.data_dir

methods = ['tf-idf', 'simple_average', 'tough_baseline', 'CoSal', 'max_pool']
f_out = open(join(DATA_PATH, "experiments", "sentences-{}.txt".format(args.exp_name)), 'w')
f_out.close()

langs = ['en', 'es', 'de', 'fr', 'fi', 'it']
embs_path = {}
embs_bin_path = {}
sent_path = {}

for lang in langs:
    embs_path[lang] = join('data', 'embs', 'wiki.{}.vec'.format(lang))
    embs_bin_path[lang] = join('data', 'embs', '{}-200000.pickle'.format(lang))

    if lang == 'en':
        sent_path[lang] = join('data', 'europarl', 'europarl-v7.de-en.en'.format(lang, lang))
    else:
        sent_path[lang] = join('data', 'europarl', 'europarl-v7.{}-en.{}'.format(lang, lang))

for k, v in embs_path.items():

    embs, word2vec, word2id, id2word = load_embs(v, 200000)
    logging.info('Loaded word embeddings for {}'.format(k))

    corpus = load_sentence_data(sent_path[k], word2id, args.max_sent)
    logging.info('Loaded sentence corpora for {}'.format(k))

    word_probs = calc_word_probs(corpus)
    logging.info('Calculated word probabilities for {}'.format(k))

    with open(join('data', "europarl", "word-probs-{}.pickle".format(k)), 'wb') as f:
        pickle.dump(word_probs, f)
    logging.info('Saved word probabilities for {}'.format(k))


def experiment(src_lang, tgt_lang, method):
    call_str = "/home/rohit/anaconda3/envs/InfoRetrieval36/bin/python "
    call_str += "unsupervised_sentences.py --src_lang {} --tgt_lang {} --method {}".format(src_lang, tgt_lang, method)
    out = subprocess.check_output(call_str.split(), stderr=subprocess.STDOUT, env=my_env, cwd=my_cwd)
    try:
        with open(join(DATA_PATH, "experiments", "sentences-{}.txt".format(args.exp_name)), 'a') as f_out:
            f_out.write('{}-{}\n'.format(src_lang, tgt_lang))
            f_out.write(out.decode('ascii'))
            f_out.write('\n')
    except subprocess.CalledProcessError as e:
        print(e.output.decode('ascii'))


for lang in langs:
    if lang == 'en':
        continue
    for method in methods:
        logging.info('Starting experiment {}, {}, {}'.format(lang, 'en', method))
        experiment(lang, 'en', method)
        logging.info('Done with experiment.')

        logging.info('Starting experiment {}, {}, {}'.format('en', lang, method))
        experiment('en', lang, method)
        logging.info('Done with experiment.')
