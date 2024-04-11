from agent import Agent
from game import Game

if __name__ == "__main__":
    game = Game(500, 500)
    agent = Agent()
    while True:
        game_over, reward, score = game.step(agent.predict_action())
