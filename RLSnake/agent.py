from game import Game
import random

class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.game = Game(500, 500)
        pass

    def remember(self, state, action, reward, next_state, done):
        # deque?
        pass

    def get_state(self, game):
        state = [game.snake.x, game.snake.y, game.snake.foodx, game.snake.foody] # later state probably will contain walls' and borders' coords also
        return state

    def do_action(self, snake):
        random_num=random.randint(0, 3)
        if random_num == 0:
            return [0, 0, 0, 1]
        if random_num == 1:
            return [0, 0, 1, 0]
        if random_num == 2:
            return [0, 1, 0, 0]
        if random_num == 3:
            return [1, 0, 0, 0]
        pass