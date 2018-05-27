from os.path import join
import logging as logging_master

from aggregation import load_sentence_data
from utils import load_embs_bin, create_padded_data

import numpy as np

DATA_PATH = "/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data"

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)

langs = ['es', 'en']

TRAINING_SIZE = 10**6
VALIDATION_SIZE = 5*10**3
TEST_SIZE = 10**4


lang2corpus = {}
for lang in langs:
    emb_path = join(DATA_PATH, "embs", "{}-200000.pickle".format(lang))
    embs, word2vec, word2id, id2word = load_embs_bin(emb_path)
    del embs, word2vec, id2word

    if lang == 'en':
        sent_path = join(DATA_PATH, "europarl", "europarl-v7.de-en.en")
    else:
        sent_path = join(DATA_PATH, "europarl", "europarl-v7.{}-en.{}".format(lang, lang))
    corpus = load_sentence_data(sent_path, word2id, TRAINING_SIZE + VALIDATION_SIZE + TEST_SIZE)
    padded_corpus = create_padded_data(corpus)
    lang2corpus[lang] = padded_corpus
    logging.info("Corpus created for {}".format(lang))


def create_files(lang1, lang2):
    training_f = join(DATA_PATH, "training", "{}-{}.training".format(lang1, lang2))
    with open(training_f, "a") as f_train:
        for idx in range(TRAINING_SIZE):
            lang1_vec = ','.join([str(x) for x in lang2corpus[lang1][idx]])
            lang2_vec_correct = ','.join([str(x) for x in lang2corpus[lang2][idx]])
            f_train.write("{}@{}\t{}@{}\t1\n".format(lang1, lang1_vec, lang2, lang2_vec_correct))
            for jdx in np.random.randint(TRAINING_SIZE, size=5):
                if jdx == idx:
                    continue
                lang2_vec_incorrect = ','.join([str(x) for x in lang2corpus[lang2][jdx]])
                f_train.write("{}@{}\t{}@{}\t-1\n".format(lang1, lang1_vec, lang2, lang2_vec_incorrect))
    logging.info("Training file done for {}-{}".format(lang1, lang2))

    validation_f = join(DATA_PATH, "training", "{}-{}.validation".format(lang1, lang2))
    with open(validation_f, "w") as f_valid:
        for idx in range(TRAINING_SIZE, TRAINING_SIZE + VALIDATION_SIZE):
            lang1_vec = ','.join([str(x) for x in lang2corpus[lang1][idx]])
            lang2_vec = ','.join([str(x) for x in lang2corpus[lang2][idx]])
            f_valid.write("{}\t{}\n".format(lang1_vec, lang2_vec))
    logging.info("Validation file done for {}-{}".format(lang1, lang2))

    testing_f = join(DATA_PATH, "training", "{}-{}.testing".format(lang1, lang2))
    with open(testing_f, "w") as f_test:
        for idx in range(TRAINING_SIZE + VALIDATION_SIZE, TRAINING_SIZE + VALIDATION_SIZE + TEST_SIZE):
            lang1_vec = ','.join([str(x) for x in lang2corpus[lang1][idx]])
            lang2_vec = ','.join([str(x) for x in lang2corpus[lang2][idx]])
            f_test.write("{}\t{}\n".format(lang1_vec, lang2_vec))
    logging.info("Testing file done for {}-{}".format(lang1, lang2))


create_files(**langs)


