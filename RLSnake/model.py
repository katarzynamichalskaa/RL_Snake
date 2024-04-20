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
        self.gamma = 0.6

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def create_model(self, state_space_size, action_space_size):
        self.model = DQN(state_space_size, action_space_size)
        self.loss_fn = nn.HuberLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        return self.model, self.loss_fn, self.optimizer

    def train_model(self, state, action, reward, next_state, done):

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

