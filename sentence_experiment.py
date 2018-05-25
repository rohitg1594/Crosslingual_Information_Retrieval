import subprocess
import os
from os.path import join
import logging as logging_master
import pickle

from utils import load_embs
from aggregation import load_sentence_data
from utils import calc_word_probs

MAX_SENTENCES = 300000

my_env = os.environ.copy()
my_env["PATH"] = "/home/rohit/anaconda3/envs/InfoRetrieval36"
my_cwd = os.path.dirname(os.path.realpath(__file__))

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)

DATA_PATH = "/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data"
methods = ['tf-idf', 'simple_average', 'tough_baseline', 'CoSal']
f_out = open(join(DATA_PATH, "experiments", "sentences-2.txt"), 'a')
langs = ['en', 'es']
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

    corpus = load_sentence_data(sent_path[k], word2id, MAX_SENTENCES)
    logging.info('Loaded sentence corpora for {}'.format(k))

    word_probs = calc_word_probs(corpus)
    logging.info('Calculated word probabilities for {}'.format(k))

    with open(join('data', "europarl", "word-probs-{}.pickle".format(k)), 'wb') as f:
        pickle.dump(word_probs, f)
    logging.info('Saved word probabilities for {}'.format(k))


def experiment(src_lang, tgt_lang, method):
    call_str = "/home/rohit/anaconda3/envs/InfoRetrieval36/bin/python "
    call_str += "sentences.py --src_lang {} --tgt_lang {} --aggr_sents {}".format(src_lang, tgt_lang, method)
    try:
        out = subprocess.check_output(call_str.split(), stderr=subprocess.STDOUT, env=my_env, cwd=my_cwd)
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
