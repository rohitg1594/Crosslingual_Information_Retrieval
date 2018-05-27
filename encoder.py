import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import logging as logging_master


logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('wikinet')
logging.setLevel(logging_master.INFO)

torch.manual_seed(1)


class Encoder(nn.Module):

    def __init__(self, lang2embs):
        super(Encoder, self).__init__()

        self.lang2embs = {}
        for lang, embs in lang2embs.items():
            vocab, dim = embs.shape
            embeddings = nn.Embedding(vocab, dim, padding_idx=0, sparse=True)
            embeddings.weight.data.copy_(torch.from_numpy(embs))
            embeddings.weight.requires_grad = False
            self.lang2embs[lang] = embeddings

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(300, 300)
        self.lin1.weight.data.copy_(torch.diag(torch.ones(300)))

        self.batn1 = nn.BatchNorm1d(300, affine=False)
        self.batn2 = nn.BatchNorm1d(300, affine=False)
        self.batn3 = nn.BatchNorm1d(400, affine=False)
        self.batn4 = nn.BatchNorm1d(100, affine=False)

        self.lin2 = nn.Linear(300, 300)
        self.lin2.weight.data.copy_(torch.diag(torch.ones(300)))
        self.lin3 = nn.Linear(901, 400)
        self.lin4 = nn.Linear(400, 100)
        self.lin5 = nn.Linear(100, 1)

    def forward(self, inputs):
        """
        """
        langs_1, sents_1, langs_2, sents_2 = inputs

        x1 = torch.zeros(sents_1.shape[0], sents_1.shape[1], 300)
        x2 = torch.zeros(sents_1.shape[0], sents_1.shape[1], 300)

        for idx, lang in enumerate(langs_1):
            x1[idx] = self.lang2embs[lang](sents_1[idx])
            x2[idx] = self.lang2embs[langs_2[idx]](sents_2[idx])

        x1 = x1.sum(dim=1).squeeze_(dim=1)
        x2 = x2.sum(dim=1).squeeze_(dim=1)

        o1 = self.relu(self.batn2(self.lin2(self.relu(self.batn1(self.lin1(x1))))))
        o2 = self.relu(self.batn2(self.lin2(self.relu(self.batn1(self.lin1(x2))))))

        cor = torch.sum(o1 * o2, 1, keepdim=True)
        nxt = torch.cat([o1, o2, torch.abs(o1 - o2), cor], 1)
        score = self.lin5(self.relu(self.batn4(self.lin4(self.relu(self.batn3(self.lin3(nxt)))))))

        prob = F.sigmoid(score)
        return prob, o1, o2

