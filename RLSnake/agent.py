from game import Game
import random
import numpy as np


class Agent:
    def __init__(self):
        self.number_of_games = 0
        pass

    def remember(self, state, action, reward, next_state, done):
        # deque?
        pass

    def get_state(self, game):
        state = [game.snake.x, game.snake.y, game.snake.foodx, game.snake.foody]    # later state probably will contain walls' and borders' coords also
        return state

    def predict_action(self):  #full random by now
        actions = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        prediction = random.choice(actions)
        return prediction