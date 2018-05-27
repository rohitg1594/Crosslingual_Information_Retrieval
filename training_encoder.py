# @author - Rohit Gupta
# @email - rogupta@mail.uni-mannheim.de
# @year -  2018

from os.path import join
import argparse
import numpy as np
import logging as logging_master
import time
import pickle

import faiss

import torch
import torch.optim as optim
import torch.nn.functional as F
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
#data_path = "/dev/shm/rogupta/info/data"
data_path = args.data_path

LOG_FILENAME = join(data_path, 'log', 'wikinet.log')
logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('count link structure of wikipedia')
logging.setLevel(logging_master.INFO)
logging.info('GPU is Available : {}'.format(use_cuda))
logging.info("DATA PATH - {}".format(args.data_path))

src_lang = 'es'
tgt_lang = 'en'

lang2embs = {}
for lang in (src_lang, tgt_lang):
    embs, _, _, _, = load_embs_bin(join(data_path, 'embs', '{}-200000.pickle'.format(lang)))
    if lang != "en":
        with open(join(data_path, "mapping", "{}-en-200000-supervised.pickle".format(lang)), 'rb') as f:
            mapper = pickle.load(f)
        embs = embs @ mapper
    embs[0] = np.zeros(300)  # ignore first row for padding
    lang2embs[lang] = embs
logging.info('word embedings loaded')


validation_f = join(data_path, "training", "{}-{}.validation".format(src_lang, tgt_lang))
src_lang_val_corpus = torch.zeros(5000, 50).type(torch.LongTensor)
tgt_lang_val_corpus = torch.zeros(5000, 50).type(torch.LongTensor)

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

        src_lang_ten[idx], tgt_lang_ten[idx] = lang1_ten, lang2_ten

src_lang_ten, tgt_lang_ten = Variable(src_lang_ten), Variable(tgt_lang_ten)


def evaluate(model):
    model.eval()

    src_lang_list = [src_lang for _ in range(5000)]
    tgt_lang_list = [tgt_lang for _ in range(5000)]


    prediction, lang1_out, lang2_out = encoder((src_lang_list, src_lang_ten, tgt_lang_list, tgt_lang_ten))
    lang1_out, lang2_out = F.normalize(lang1_out, dim=1), F.normalize(lang2_out, dim=1)
    print("Validation  Accuracy : {}".format((prediction.data.numpy() > 0.5).sum()/5000 ))
    lang1_out, lang2_out = lang1_out.data.numpy(), lang2_out.data.numpy()

    index = faiss.IndexFlatIP(300)
    index.add(lang2_out.astype(np.float32))
    D_ip, I_ip = index.search(lang1_out.astype(np.float32), 20)

    logging.info("validation results for language pair - inner-product - {}-{}".format(src_lang, tgt_lang))
    eval_sents(I_ip, [1, 5, 10])

    index = faiss.IndexFlatL2(300)
    index.add(lang2_out.astype(np.float32))
    D_l2, I_l2 = index.search(lang1_out.astype(np.float32), 20)

    logging.info("validation results for language pair - L2 - {}-{}".format(src_lang, tgt_lang))
    eval_sents(I_l2, [1, 5, 10])


train_data = EncodingDataset(join(data_path, "training", "{}-{}.training".format(src_lang, tgt_lang)))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=args.n)

logging.info('data loader created')

encoder = Encoder(lang2embs)

if use_cuda:
    encoder = encoder.cuda()

optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, encoder.parameters()))

loss_fn = torch.nn.BCELoss()

losses = []
logging.info('starting training')
cycle_i = 0
for epoch in range(20):

    running_loss = 0.0
    tic = time.time()
    for i, data in enumerate(train_loader, 0):
        encoder.train()

        lang1, lang1_sents, lang2, lang2_sents, label = data
        optimizer.zero_grad()
        prediction, _, _ = encoder((lang1, lang1_sents, lang2, lang2_sents))
        loss = loss_fn(prediction, label)

        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        losses.append(loss.data[0])

        if i % 1000 == 0:
            logging.info('Epoch - {}, iter - {} - Loss - {}'.format(epoch, i, loss.data[0]))
        if i % 25000 == 0:
            evaluate(encoder)

    torch.save(encoder.state_dict(), join(data_path, 'models', "encoder-cosine2-{}-comp".format(epoch)))

print('Finished Training')