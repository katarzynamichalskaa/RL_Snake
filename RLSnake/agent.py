import random
from collections import deque
import torch
from model import DQN
from trainer import Trainer
from snake import Directions
from utils import PATH


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.max_memory = 100_000
        self.memory = deque(maxlen=self.max_memory)
        self.BATCH_SIZE = 1000
        self.lr = 0.001
        self.gamma = 0.9
        self.model = DQN(n_observations=22, n_actions=4).to('cuda' if torch.cuda.is_available() else 'cpu')
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

    @staticmethod
    def get_state(game):
        # borders
        danger_left, danger_up, danger_right, danger_down = game.snake.borders_danger(offset=2)

        # segment_borders
        s_danger_left, s_danger_up, s_danger_right, s_danger_down = game.snake.segment_danger(offset=7)

        # wall_borders
        wall_left, wall_right, wall_up, wall_down = game.snake.wall_danger(offset=7)

        # food loc
        food_left = int(game.snake.x < game.snake.food_x)
        food_right = int(game.snake.x > game.snake.food_x)
        food_up = int(game.snake.y < game.snake.food_y)
        food_down = int(game.snake.y > game.snake.food_y)

        perfect_x, perfect_y = game.snake.perfect_line(wall_left, wall_up, wall_right, wall_down)

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
                 danger_left, danger_up, danger_right, danger_down,
                 s_danger_left, s_danger_up, s_danger_right, s_danger_down,
                 wall_left, wall_right, wall_up, wall_down]
                 # *game.snake.map_around()]
        return state

    @staticmethod
    def random_action():
        actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return random.choice(actions)

    def save_model(self):
        torch.save(self.model, PATH)

    @staticmethod
    def load_model():
        model = torch.load(PATH)
        model.eval()
        return model





