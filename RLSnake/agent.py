from game import Game


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
        pass

    def learn(self):
        pass