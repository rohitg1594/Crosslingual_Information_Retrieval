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
np.set_printoptions(threshold=1000)

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval - Learn Word Embedding Mapping",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# data path
parser.add_argument("--data_path",
                    default="/home/rohit/Documents/Spring_2018/Information_retrieval/Project/Crosslingual_Information_Retrieval/data",
                    help="data path")
parser.add_argument("--n", default=8, type=int, help="num_cores")

args = parser.parse_args()



use_cuda = torch.cuda.is_available()
data_path = "/dev/shm/rogupta/info/data"

LOG_FILENAME = join(data_path, 'log', 'wikinet.log')
logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('count link structure of wikipedia')
logging.setLevel(logging_master.INFO)
logging.info('GPU is Available : {}'.format(use_cuda))
logging.info("DATA PATH - {}".format(args.data_path))

langs = ['es', 'de', 'fr', 'fi', 'it', 'en']
lang2embs = {}
for lang in langs:
    embs, _, _, _, = load_embs_bin(join(data_path, 'embs', '{}-200000.pickle'.format(lang)))
    if lang != "en":
        with open(join(data_path, "mapping", "{}-en-200000-supervised.pickle".format(lang)), 'rb') as f:
            mapper = pickle.load(f)
        embs = embs @ mapper
    lang2embs[lang] = embs
logging.info('word embedings loaded')

lang_pairs = set()
for lang in langs:
    if lang == 'en':
        continue
    lang_pairs.add((lang, 'en'))
    lang_pairs.add(('en', lang))

for l in lang_pairs:
    print(l)

validation_sets = {}
for lang_pair in lang_pairs:
    lang1, lang2 = lang_pair
    validation_f = join(data_path, "training", "{}-{}.validation".format(lang1, lang2))
    i1 = torch.zeros(5000, 50).type(torch.LongTensor)
    i2 = torch.zeros(5000, 50).type(torch.LongTensor)

    with open(validation_f, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            try:
                parts = line.split('\t')
                lang1_str = parts[0]
                lang2_str = parts[1]
            except ValueError:
                continue

            lang1_vec = np.fromstring(lang1_str, sep=',')
            lang2_vec = np.fromstring(lang2_str, sep=',')
            lang1_ten = torch.from_numpy(lang1_vec)
            lang2_ten = torch.from_numpy(lang2_vec)
            lang1_ten, lang2_ten = lang1_ten.type(torch.LongTensor), lang2_ten.type(torch.LongTensor)

            i1[idx], i2[idx] = lang1_ten, lang2_ten

    validation_sets[lang_pair] = Variable(i1), Variable(i2)



def evaluate(model):
    model.eval()
    eval_lang_pairs = random.sample(lang_pairs, 2)
    for lang_pair in eval_lang_pairs:
        lang1, lang2 = lang_pair
        sents_1, sents_2 = validation_sets[lang_pair]

        lang1_list = [lang1 for _ in range(5000)]
        lang2_list = [lang2 for _ in range(5000)]

        lang1_out, lang2_out = encoder((lang1_list, sents_1)), encoder((lang2_list, sents_2))
        lang1_out, lang2_out = lang1_out.data.numpy(), lang2_out.data.numpy()

        index = faiss.IndexFlatIP(300)
        index.add(lang2_out.astype(np.float32))
        D, I = index.search(lang1_out.astype(np.float32), 20)

        logging.info("validation results for language pair - {}".format(lang_pair))
        eval_sents(I, [1, 5, 10])


train_data = EncodingDataset(join(data_path, "training", "new-training.tsv"))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=args.n)

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
        if i % 10000 == 0:
            evaluate(encoder)

    torch.save(encoder
               .state_dict(), join(data_path, 'models', "encoder-{}-comp".format(epoch)))

print('Finished Training')