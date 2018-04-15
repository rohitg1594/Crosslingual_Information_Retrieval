import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Trainer():
    def __init__(self, optim_fn, src_embs, tgt_embs, batch_size, smooth, discriminator, mapper):
        self.optim_fn = optim_fn
        self.src_embs = src_embs.astype(np.float32)
        self.tgt_embs = tgt_embs.astype(np.float32)
        self.batch_size = batch_size
        self.smooth = smooth
        self.discriminator = discriminator
        self.mapper = mapper
        self.dis_optimizer = optim_fn(discriminator.parameters())
        self.map_optimizer = optim_fn(mapper.parameters())


    def _get_train_batch(self):
        '''Get input training data'''
        mask = np.random.choice(self.src_embs.shape[0], size=self.batch_size)
        x_src = torch.from_numpy(self.src_embs[mask])
        x_tgt = torch.from_numpy(self.tgt_embs[mask])
        X = Variable(torch.cat((x_src, x_tgt), 0))
        Y = Variable(torch.zeros(2*self.batch_size))
        Y[:self.batch_size] = 1 - self.smooth
        Y[self.batch_size] = self.smooth

        return X, Y


    def dis_step(self):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self._get_train_batch()
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)


        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()

        return loss.data.numpy()


    def mapping_step(self):
        """
        Mapping training step.
        """
        self.discriminator.eval()

        # loss
        x, y = self._get_train_batch()
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()

        return loss.data.numpy()
