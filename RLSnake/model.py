import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):   # from pytorch documentation

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def create_model(self, state_space_size, action_space_size):
        model = DQN(state_space_size, action_space_size)
        loss_fn = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters())
        return model, loss_fn, optimizer


class Trainer:
    def __init__(self):
        pass