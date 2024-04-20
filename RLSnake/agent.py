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
        danger_left, danger_up, danger_right, danger_down = game.snake.check_danger(2)

        state = [game.snake.x, game.snake.y, game.snake.foodx, game.snake.foody,
                 danger_left, danger_up, danger_right,
                 danger_down]
        return state

    def predict_action(self):  # full random by now
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        prediction = random.choice(actions)
        return prediction
