import argparse
import numpy as np
import os
import pickle
import logging as logging_master
import sys

import faiss

import torch.optim as optim
import torch.nn as nn
import torch

from sklearn.cross_decomposition import CCA

from utils import load_embs, load_embs_bin, load_dictionary, get_parallel_data, str2bool, procrustes
from evaluation import eval_main, eval_w
from discriminator import Discriminator
from trainer import Trainer

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
# Unsupervised training
parser.add_argument("--hidden_dim", default="1024", type=int, help="dimension of the discriminator hidden layer")
parser.add_argument("--num_hidden", default="2", type=int, help="number of hidden layers in the discriminator")
parser.add_argument("--input_do", default="0", type=int, help="amount of hidden dropout")
parser.add_argument("--hidden_do", default="0.1", type=float, help="amount of hidden layer dropout")
parser.add_argument("--smooth", default="0.1", type=float, help="smoothing parameter for adversarial training")
parser.add_argument("--batch_size", default="128", type=int, help="batch size")
parser.add_argument("--num_epochs", default="10", type=int, help="number of epochs for adversarial training")
parser.add_argument("--evaluate_every", default="10", type=int, help="number of epochs after which to evaluate")
# Word embedding method choice
parser.add_argument("--method", choices=['procrustes', 'unsupervised', 'CCA'],
                    default="procrustes", help="method to learn word embeddings")
# Export
parser.add_argument("--export", default=True, type=str2bool, help="whether to export learned mapping matrix")


args = parser.parse_args()

for k, v in vars(args).items():
    print('{:<30}\t{}'.format(k, v))

beta = args.beta
max_vocab = args.max_vocab
norm = args.norm
emb_dim = args.emb_dim
smooth = args.smooth
batch_size = args.batch_size
num_epochs = args.num_epochs
hidden_dim = args.hidden_dim
num_hidden = args.num_hidden
input_do = args.input_do
hidden_do = args.hidden_do
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
train_dict = os.path.join(args.data_dir, "dictionaries", args.src_lang + '-' + args.tgt_lang + '.0-5000.txt')
test_dict = os.path.join(args.data_dir, "dictionaries", args.src_lang + '-' + args.tgt_lang + '.5000-6500.txt')
assert os.path.exists(train_dict)
assert os.path.exists(test_dict)


if args.save_pickle:
    logging.info("saving loaded embeddings in pickle dump")
    with open(os.path.join(args.data_dir, "embs", '{}-{}.pickle'.format(args.src_lang, args.max_vocab)), 'wb') as f:
        pickle.dump((src_embs, src_word2vec, src_word2id, src_id2word), f)
    with open(os.path.join(args.data_dir, "embs", '{}-{}.pickle'.format(args.tgt_lang, args.max_vocab)), 'wb') as f:
        pickle.dump((tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word), f)
    logging.info("embeddings saved in pickle dump")


if args.method == 'procrustes':
    dico = load_dictionary(train_dict, -1, src_word2id, tgt_word2id)
    X, Y = get_parallel_data(src_embs, tgt_embs, dico)
    W = procrustes(X, Y)

    if int(args.ortho):
        W = (1 + beta)*W - beta*(W@W.T)@W

elif args.method == 'unsupervised':

    optimizer = optim.Adam

    # Setup the discriminator and the mapper
    discriminator = Discriminator(emb_dim, hidden_dim, num_hidden, input_do, hidden_do)
    mapper = nn.Linear(emb_dim, emb_dim)
    mapper.weight.data.copy_(torch.diag(torch.ones(emb_dim)))

    # Change the numpy embeddings to torch embeddings
    src_embs_torch = nn.Embedding(len(src_embs), emb_dim)
    src_embs_torch.weight.data.copy_(torch.from_numpy(src_embs.astype(np.float32)))
    tgt_embs_torch = nn.Embedding(len(tgt_embs), emb_dim)
    tgt_embs_torch.weight.data.copy_(torch.from_numpy(tgt_embs.astype(np.float32)))

    # We don't want to train the embeddings
    src_embs_torch.weight.requires_grad = False
    tgt_embs_torch.weight.requires_grad = False

    # Setup the trainer
    trainer = Trainer(optimizer, src_embs_torch, tgt_embs_torch, batch_size, smooth, discriminator, mapper, beta)

    # Training loop
    for i in range(num_epochs):
        num_iters = 0
        batch = 0
        while num_iters <= len(src_embs):
            for j in range(5):
                dis_loss = trainer.dis_step()
            mapper_loss = trainer.mapping_step()

            num_iters += batch_size
            batch += 1

            if batch % 10 == 0:
                logging.info("Epoch : {}, iter : {}, Disciminator Loss: {}, Mapper Loss : {}".format(i, batch,
                                                                                                     dis_loss,
                                                                                                     mapper_loss))
            if batch % evaluate_every == 0:

                W = mapper.weight.data.numpy()
                eval_main(W, test_dict, src_word2id, tgt_word2id, src_embs, tgt_embs, src_id2word, tgt_id2word,
                          verbose=False)

elif args.method == 'CCA':
    dico_train = load_dictionary(train_dict, -1, src_word2id, tgt_word2id)
    dico_test = load_dictionary(test_dict, -1, src_word2id, tgt_word2id)

    X_train, Y_train = get_parallel_data(src_embs, tgt_embs, dico_train)
    X_test, Y_test = get_parallel_data(src_embs, tgt_embs, dico_test)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    num_components = 80
    cca = CCA(n_components=num_components)
    logging.info("learning cca transformation")
    cca.fit(X_train, Y_train)
    logging.info("cca transformation learned")

    X_test_c, Y_test_c = cca.transform(X_test, Y_test)

    index = faiss.IndexFlatIP(num_components)
    index.add(Y_test_c.astype(np.float32))
    d, i = index.search(X_test_c.astype(np.float32), 20)

    dicts = eval_w(i, dico_test, src_id2word)
    sys.exit(1)





logging.info('mapping matrix learned')
logging.info('evaluating final mapping')
eval_main(W, test_dict, src_word2id, tgt_word2id, src_embs, tgt_embs, src_id2word, tgt_id2word, verbose=False)

if args.export:
    f_name = os.path.join(args.data_dir, "mapping", '{}-{}-{}-{}.pickle'.format(args.src_lang, args.tgt_lang,
                                                                                args.max_vocab, args.method))
    logging.info('saving learned mapping to {}'.format(f_name))
    with open(f_name, 'wb') as f:
        pickle.dump(W, f)
