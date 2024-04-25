import random
import numpy as np
from collections import deque
from model import DQN, Trainer
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

        # borders
        danger_left, danger_up, danger_right, danger_down = game.snake.check_danger(2)

        # food loc
        food_left = game.snake.x < game.snake.foodx
        food_right = game.snake.x > game.snake.foodx
        food_up = game.snake.y < game.snake.foody
        food_down = game.snake.y > game.snake.foody
        perfect_x = game.snake.x == game.snake.foodx
        perfect_y = game.snake.y == game.snake.foody

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
                 danger_left, danger_up, danger_right, danger_down,
                 dir_left, dir_right, dir_up, dir_down]

        return np.array(state, dtype=int)

    def random_action(self):
        actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return random.choice(actions)
