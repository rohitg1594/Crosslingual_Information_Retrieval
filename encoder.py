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
            self.embeddings = nn.Embedding(vocab, dim, padding_idx=0, sparse=True)
            self.embeddings.weight.data.copy_(torch.from_numpy(embs))
            self.embeddings.weight.requires_grad = False

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
        lang, batch = inputs
        try:
            embs = self.lang2embs[lang](batch)
        except RuntimeError as e:
            print(batch, e)

        output = F.normalize(embs.sum(dim=1), dim=1)

        output = self.relu(self.batn2(self.lin2(self.relu(self.batn1(self.lin1(output))))))

        return output

