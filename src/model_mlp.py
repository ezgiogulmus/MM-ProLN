import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, config, hidden_dim=1024):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config["cli_size"], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = True if config["class_weight"] is None else False

    def forward(self, inputs):
        x = inputs[-1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.sigmoid:
            return torch.sigmoid(x)
        return x