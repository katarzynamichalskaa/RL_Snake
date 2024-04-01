from game import Game


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.game = Game(500, 500)
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def get_state(self, snake):
        #return state
        pass

    def do_action(self, snake):
        pass

    def learn(self):
        pass