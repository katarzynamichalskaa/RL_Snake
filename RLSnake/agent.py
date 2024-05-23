import random
import numpy as np
from collections import deque

import torch

from model import DQN
from trainer import Trainer
from snake import Directions


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.max_memory = 100_000
        self.memory = deque(maxlen=self.max_memory)
        self.BATCH_SIZE = 1000
        self.lr = 0.001
        self.gamma = 0.9
        self.model = DQN(n_observations=14, n_actions=4)
        self.trainer = Trainer(self.model, lr=self.lr, gamma=self.gamma)
        self.PATH = 'model/snake_model.pth'

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_model(states, actions, rewards, next_states, dones)

    def get_state(self, game):
        # danger info
        l, u, r, d = self.join_danger_info(game)

        # food loc
        food_left = game.snake.x < game.snake.food_x
        food_right = game.snake.x > game.snake.food_x
        food_up = game.snake.y < game.snake.food_y
        food_down = game.snake.y > game.snake.food_y
        perfect_x = game.snake.x == game.snake.food_x
        perfect_y = game.snake.y == game.snake.food_y

        # snake direction
        direction = game.snake.direction

        dir_left = 0
        dir_right = 0
        dir_up = 0
        dir_down = 0

        if direction == Directions.LEFT:
            dir_left = 1
        if direction == Directions.RIGHT:
            dir_right = 1
        if direction == Directions.UP:
            dir_up = 1
        if direction == Directions.DOWN:
            dir_down = 1

        # full state
        state = [food_right, food_left, food_up, food_down, perfect_x, perfect_y,
                 dir_left, dir_right, dir_up, dir_down,
                 l, u, r, d]

        return np.array(state, dtype=int)

    @staticmethod
    def random_action():
        actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return random.choice(actions)

    @staticmethod
    def join_danger_info(game):
        dangerous_info = [0, 0, 0, 0]

        # borders
        danger_left, danger_up, danger_right, danger_down = game.snake.check_danger(offset=2)

        # segment_borders
        s_danger_left, s_danger_up, s_danger_right, s_danger_down = game.snake.segment_danger(offset=7)

        # wall_borders
        w_danger_left, w_danger_up, w_danger_right, w_danger_down = game.snake.wall_danger(offset=7)

        if danger_left == 1 or s_danger_left or w_danger_left == 1:
            dangerous_info[0] = 1
        if danger_up == 1 or s_danger_up or w_danger_up == 1:
            dangerous_info[1] = 1
        if danger_right == 1 or s_danger_right or w_danger_right == 1:
            dangerous_info[2] = 1
        if danger_down == 1 or s_danger_down or w_danger_down == 1:
            dangerous_info[3] = 1

        return dangerous_info

    def save_model(self):
        torch.save(self.model, self.PATH)

    def load_model(self):
        model = torch.load(self.PATH)
        model.eval()
        return model





