import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from utils import read_emb

dtype = torch.FloatTensor


class Discriminator(nn.Module):
    '''Discriminator NN for adversarial training'''
    def __init__(self, input_dim, hidden_dim, num_hidden, input_DO, hidden_DO):
        super(Discriminator, self).__init__()

        layers = [nn.Dropout(input_DO)]

        for i in range(num_hidden + 1):
            if i == 0:
                id = input_dim
                od = hidden_dim
            elif i == num_hidden:
                id = hidden_dim
                od = 1
            else:
                id = od = hidden_dim
            layers.append(nn.Linear(id, od))

            if i < num_hidden:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(hidden_DO))

        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, X):
        return self.layers(X).view(-1)

discriminator = Discriminator(300, 100, 2, 0, 0)


def train():
    

def get_input_batch(batch_size, emb_matr):
    word2vec, word2id, id2word = read_emb(emb_path)
    id_mask = np.random.choice(len(word2vec), size=batch_size)
    x = np.zeros(batch_size, emb_size)
    for i in id_mask:
        x[i] = word2vec[id2word[i]]

    

def dis_step(self, stats):
    """
    Train the discriminator.
    """
    self.discriminator.train()
    
    # loss
    x, y = self.get_dis_xy(volatile=True)
    preds = self.discriminator(Variable(x.data))
    loss = F.binary_cross_entropy(preds, y)
    stats['DIS_COSTS'].append(loss.data[0])
    
    # check NaN
    if (loss != loss).data.any():
        logger.error("NaN detected (discriminator)")
        exit()
        
    # optim
    self.dis_optimizer.zero_grad()
    loss.backward()
    self.dis_optimizer.step()
    clip_parameters(self.discriminator, self.params.dis_clip_weights)

def mapping_step(self, stats):
    """
    Fooling discriminator training step.
    """
    if self.params.dis_lambda == 0:
        return 0

    self.discriminator.eval()
    
    # loss
    x, y = self.get_dis_xy(volatile=False)
    preds = self.discriminator(x)
    loss = F.binary_cross_entropy(preds, 1 - y)
    loss = self.params.dis_lambda * loss
    
    # check NaN
    if (loss != loss).data.any():
        logger.error("NaN detected (fool discriminator)")
        exit()
        
    # optim
    self.map_optimizer.zero_grad()
    loss.backward()
    self.map_optimizer.step()
    self.orthogonalize()
    
    return 2 * self.params.batch_size


