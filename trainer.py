import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Trainer():
    def __init__(self, optim_fn, src_embs, tgt_embs, batch_size, smooth, discriminator, mapper):
        self.optim_fn = optim_fn
        self.src_embs = src_embs
        self.tgt_embs = tgt_embs
        self.batch_size = batch_size
        self.smooth = smooth
        self.discriminator = discriminator
        self.mapper = mapper
        self.dis_optimizer = optim_fn(discriminator.parameters())
        self.map_optimizer = optim_fn(mapper.parameters())


    def _get_train_batch(self):
        '''Get input training data'''
        mask_src = torch.LongTensor(self.batch_size).random_(len(self.src_embs.weight))
        mask_tgt = torch.LongTensor(self.batch_size).random_(len(self.tgt_embs.weight))

        x_src = self.src_embs(Variable(mask_src))
        x_src = self.mapper(Variable(x_src.data))

        x_tgt = self.tgt_embs(Variable(mask_tgt))
        X = Variable(torch.cat([x_src.data, x_tgt.data], 0))

        Y = torch.zeros(2*self.batch_size)
        Y[:self.batch_size] = 1 - self.smooth
        Y[self.batch_size:] = self.smooth
        Y = Variable(Y)

        return X, Y


    def dis_step(self):
        """
        Train the discriminator.
        """
        self.discriminator.train()
        
        x, y = self._get_train_batch()
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)

        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()

        return loss.data.numpy()


    def mapping_step(self):
        """
        Mapping training step.
        """
        self.discriminator.eval()

        x, y = self._get_train_batch()
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)

        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()

        return loss.data.numpy()
