# @author - Rohit Gupta
# @email - rogupta@mail.uni-mannheim.de
# @year -  2018

from os.path import join
import argparse
import numpy as np
import logging as logging_master
import time
import pickle
import random

import faiss

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import load_embs_bin
from evaluation import eval_sents

from dataset import EncodingDataset
from encoder import Encoder

torch.set_printoptions(threshold=1000)

use_cuda = torch.cuda.is_available()
DATA_PATH = "/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data"

data_path = "/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data"
LOG_FILENAME = join(data_path, 'log', 'wikinet.log')
logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('count link structure of wikipedia')
logging.setLevel(logging_master.INFO)
logging.info('GPU is Available : {}'.format(use_cuda))

langs = ['es', 'de', 'fr', 'fi', 'it', 'en']
lang2embs = {}
for lang in langs:
    embs, _, _, _, = load_embs_bin(join(data_path, 'embs', '{}-200000.pickle'.format(lang)))
    if lang != "en":
        with open(join(DATA_PATH, "mapping", "{}-en-200000-supervised.pickle".format(lang)), 'rb') as f:
            mapper = pickle.load(f)
        embs = embs @ mapper
    lang2embs[lang] = embs


lang_pairs = set()
for lang1 in langs:
    for lang2 in langs:
        if lang1 == lang2:
            continue
        if (lang1, lang2) not in lang_pairs and (lang2, lang1) not in lang_pairs:
            lang_pairs.add((lang1, lang2))

validation_sets = {}
for lang_pair in lang_pairs:
    lang1, lang2 = lang_pair
    validation_f = join(DATA_PATH, "training", "{}-{}.validation".format(lang1, lang2))
    i1 = np.zeros((5000, 50), dtype=np.int32)
    i2 = np.zeros((5000, 50), dtype=np.int32)

    with open(validation_f, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            try:
                target, source = line.split('\t')
            except ValueError:
                continue

            i1[idx] = np.fromstring(source, sep=',', dtype=np.int)
            i2[idx] = np.fromstring(target, sep=',', dtype=np.int)

    validation_sets[lang_pair] = i1, i2


def evaluate(model):
    model.eval()
    eval_lang_pairs = random.sample(lang_pairs, 2)
    bs = 32
    for lang_pair in eval_lang_pairs:
        lang1, lang2 = lang_pair
        i1, i2 = validation_sets[lang_pair]

        i1 = Variable(torch.from_numpy(i1))
        i2 = Variable(torch.from_numpy(i2))

        lang1_out, lang2_out = encoder((lang1, i1)), encoder((lang2, i2))
        lang1_out, lang2_out = lang1_out.numpy(), lang2_out.numpy()

        index = faiss.IndexFlatIP(300)
        index.add(lang2_out.astype(np.float32))
        D, I = index.search(lang1_out.astype(np.float32), 20)

        logging.info("validation results for language pair - {}".format(lang_pair))
        eval_sents(I, [1, 5, 10])


logging.info('embedding matrices created')
#data_path = "/dev/shm/rogupta/data"

train_data = EncodingDataset(join(data_path, "training", "training.tsv"))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

logging.info('data loader created')

encoder = Encoder(lang2embs)

if use_cuda:
    wikinet = encoder.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()))

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

losses = []
logging.info('starting training')
cycle_i = 0
for epoch in range(20):

    running_loss = 0.0
    tic = time.time()
    for i, data in enumerate(train_loader, 0):

        lang1, lang1_sents, lang2, lang2_sents = data
        optimizer.zero_grad()
        lang1_out, lang2_out = encoder((lang1, lang1_sents)), encoder((lang2, lang2_sents))

        loss = mse_loss(lang1_out, lang2_out)

        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        losses.append(loss.data[0])

        if i % 1000 == 0:
            logging.info('Epoch - {}, iter - {} - Loss - {}'.format(epoch, i, loss.data[0]))
        # if i % 5000 == 0:
        #     evaluate(encoder)

    torch.save(encoder
               .state_dict(), join(data_path, 'models', "to-en-{}-comp".format(epoch)))

print('Finished Training')