import torch
import torch.nn as nn

class GatedActivation(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(GatedActivation, self).__init__()
        self.act = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.act(x) * self.sigmoid(x)