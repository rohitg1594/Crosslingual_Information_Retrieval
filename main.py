from utils import load_embs, load_dictionary, get_parallel_data
from evaluation import evaluation1, evaluation2, eval_data
from procrustes import procrustes
import argparse
import faiss
import numpy as np
from trainer import Trainer
import torch.optim as optim
import torch.nn as nn
import torch
from models import Discriminator


parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval")
parser.add_argument("--src_embs", default="../data/wiki.en.vec", help="File path of source embeddings")
parser.add_argument("--tgt_embs", default="../data/wiki.de.vec", help="File path of target embeddings")
parser.add_argument("--emb_dim", default="300", help="dimnsion of embeddings")
parser.add_argument("--train_dict", default="../MUSE/data/crosslingual/dictionaries/en-de.0-5000.txt", help="File path of training dictionary")
parser.add_argument("--test_dict", default="../MUSE/data/crosslingual/dictionaries/en-de.5000-6500.txt", help="File path of test dictionary")
parser.add_argument("--max_vocab", default=200000, help="Maximum vocabulary size loaded from embeddings")
parser.add_argument("--orthognalize", default=0, help="Whether to orthognolize the mapping matrix")
parser.add_argument("--beta", default=0.01, help="Beta parameter for orthognalization")
parser.add_argument("--norm", default=1, help="Normalize embeddings")
parser.add_argument("--method", default="supervised", help="supervised of unsupervised")
# Unsupervised training
parser.add_argument("--optim", default="Adam", help="optimizer function to use for supervised training")
parser.add_argument("--hidden_dim", default="1024", help="dimension of the discriminator hidden layer")
parser.add_argument("--num_hidden", default="2", help="number of hidden layers in the discriminator")
parser.add_argument("--input_do", default="0", help="amount of hidden dropout")
parser.add_argument("--hidden_do", default="0.1", help="amount of hidden layer dropout")
parser.add_argument("--smooth", default="0.1", help="smoothing parameter for adversarial training")
parser.add_argument("--batch_size", default="128", help="batch size")
parser.add_argument("--num_epochs", default="10", help="number of epochs for adversarial training")
parser.add_argument("--evaluate_every", default="10", help="number of epochs after which to evaluate")

args = parser.parse_args()
beta = float(args.beta)
max_vocab = int(args.max_vocab)
norm = int(args.norm)
emb_dim = int(args.emb_dim)
smooth = float(args.smooth)
batch_size = int(args.batch_size)
num_epochs = int(args.num_epochs)
hidden_dim = int(args.hidden_dim)
num_hidden = int(args.num_hidden)
input_do = float(args.input_do)
hidden_do = float(args.hidden_do)
evaluate_every = int(args.evaluate_every)

assert int(args.orthognalize) in [0, 1]

src_embs, src_word2vec, src_word2id, src_id2word = load_embs(args.src_embs, max_vocab, norm)
print('Loaded source embeddings')
tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = load_embs(args.tgt_embs, max_vocab, norm)
print('Loaded target embeddings')

def evaluation(W):
   dico_test = load_dictionary(args.test_dict, -1, src_word2id, tgt_word2id)
   X_test, Y_test = get_parallel_data(src_embs, tgt_embs, dico_test)

   I_test = eval_data(W, X_test, tgt_embs)

   evaluation1(I_test, dico_test)
   dicts = evaluation2(I_test, dico_test, src_id2word)
   incorrect_1 = [k for k, v in dicts[0].items() if v==0]

   for i in range(20):
       src_word = src_id2word[dico_test[i,0]]
       correct_trans = tgt_id2word[dico_test[i,1]]
       if src_word in incorrect_1:
           preds = ''
           for k in range(10):
               pred = tgt_id2word[I_test[i, k]]
               preds += pred + ', '
           preds = preds[:-2]
           print('{:<15}|{:<15}|{}'.format(src_word, correct_trans, preds))

if args.method == 'supervised':
   dico = load_dictionary(args.train_dict, -1, src_word2id, tgt_word2id)
   X, Y = get_parallel_data(src_embs, tgt_embs, dico)
   W = procrustes(X, Y)

   if int(args.orthognalize):
       W = (1 + beta)*W - beta*(W@W.T)@W

   evaluation(W)


elif args.method == 'unsupervised':

   # Get the optimizer function
    if args.optim == "Adam":
        optimizer = optim.Adam
    elif args.optim_fn == "RMSprop":
        optimizer = optim.RMSprop

    # Setup the discriminator and the mapper
    discriminator = Discriminator(emb_dim, hidden_dim, num_hidden, input_do, hidden_do)
    mapper = nn.Linear(emb_dim, emb_dim)
    mapper.weight.data.copy_(torch.diag(torch.ones(emb_dim)))

    # Change the numpy embeddings to torch embeddings
    src_embs_torch = nn.Embedding(len(src_embs), emb_dim)
    src_embs_torch.weight.data.copy_(torch.from_numpy(src_embs.astype(np.float32)))
    tgt_embs_torch = nn.Embedding(len(tgt_embs), emb_dim)
    tgt_embs_torch.weight.data.copy_(torch.from_numpy(tgt_embs.astype(np.float32)))

    # Setup the trainer
    trainer = Trainer(optimizer, src_embs_torch, tgt_embs_torch, batch_size, smooth, discriminator, mapper)

    for i in range(num_epochs):
        num_iters = 0
        while num_iters <= len(src_embs):
            for j in range(5):
                dis_loss = trainer.dis_step()
            mapper_loss = trainer.mapping_step()

            num_iters += batch_size

        print("Epoch : {}, Disciminator Loss: {}, Mapper Loss : {}".format(i, dis_loss, mapper_loss))
        if i%evaluate_every  == 0:
           W = mapper.weight.data.numpy()
           evaluation(W)
