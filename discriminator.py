from torch import nn


class Discriminator(nn.Module):
    """
    Discriminator NN for adversarial training.
    Simple MLP with with one hidden layer and dropout.
    """
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

    def forward(self, X):
        return self.layers(X).view(-1)




