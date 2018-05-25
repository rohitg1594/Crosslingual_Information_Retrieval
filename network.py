import keras
import pickle
from os.path import join
from keras.utils.vis_utils import plot_model
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize
from keras.preprocessing.text import text_to_word_sequence
import logging as logging_master

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)

DATA_PATH = "/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data/"
SENT_LIMIT = 300000

model = keras.models.load_model(join(DATA_PATH, 'models', '2018-5-22_en-es_Nadam_ratio-3_no-fooling_cp'))
print(model.layers)