import subprocess
import os
from os.path import join
import logging as logging_master

my_env = os.environ.copy()
my_env["PATH"] = "/home/rohit/anaconda3/envs/InfoRetrieval36"
my_cwd = os.path.dirname(os.path.realpath(__file__))

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)

DATA_PATH = "/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data"
methods = ['tf-idf', 'simple_average', 'tough_baseline', 'CoSal']
langs = ['es', 'fr', 'de', 'it', 'fi']
f_out = open(join(DATA_PATH, "experiments", "sentences.txt"), 'a')


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
    for method in methods:
        logging.info('Starting experiment {}, {}, {}'.format(lang, 'en', method))
        experiment(lang, 'en', method)
        logging.info('Done with experiment.')

        logging.info('Starting experiment {}, {}, {}'.format('en', lang, method))
        experiment('en', lang, method)
        logging.info('Done with experiment.')
