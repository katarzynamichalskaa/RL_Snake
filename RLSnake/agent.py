from game import Game
import random
import numpy as np
from collections import deque
from model import DQN, Trainer


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.max_memory = 100_000
        self.memory = deque(maxlen=self.max_memory)
        self.BATCH_SIZE = 1000
        self.lr = 0.001
        self.gamma = 0.9
        self.model = DQN(n_observations=8, n_actions=3)
        self.trainer = Trainer(self.model, lr=self.lr, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_model(states, actions, rewards, next_states, dones)

    def get_state(self, game):
        danger_left, danger_up, danger_right, danger_down = game.snake.check_danger(2)

        state = [game.snake.x, game.snake.y, game.snake.foodx, game.snake.foody,
                 danger_left, danger_up, danger_right,
                 danger_down]
        return state

    def predict_action(self):
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        return random.choice(actions)
