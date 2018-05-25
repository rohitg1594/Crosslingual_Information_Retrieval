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

        self.batn1 = nn.BatchNorm1d(300, affine=False)
        self.batn2 = nn.BatchNorm1d(300, affine=False)
        self.batn3 = nn.BatchNorm1d(400, affine=False)
        self.batn4 = nn.BatchNorm1d(100, affine=False)

        self.lin2 = nn.Linear(300, 300)
        self.lin3 = nn.Linear(901, 400)
        self.lin4 = nn.Linear(400, 100)
        self.lin5 = nn.Linear(100, 1)

    def forward(self, inputs):
        """
        """
        langs, sents = inputs
        try:
            embs = torch.zeros(sents.shape[0], sents.shape[1], 300)
        except AttributeError as e:
            print('here', e)
        try:
            for idx, lang in enumerate(langs):
                embs[idx] = self.lang2embs[lang](sents[idx])
        except RuntimeError as e:
            print(langs, sents, e)

        output = F.normalize(embs.sum(dim=1), dim=1)

        output = self.relu(self.batn2(self.lin2(self.relu(self.batn1(self.lin1(output))))))

        output = F.normalize(output, dim=1)

        return output

