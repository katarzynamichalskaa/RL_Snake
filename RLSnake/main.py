from agent import Agent
from game import Game
from model import DQN
import numpy as np
import torch

game = Game(100, 100)
agent = Agent()
model = DQN(4, 4)
model, loss_fn, optim = model.create_model(4, 4)
alpha = 0.5
gamma = 0.6
epsilon = 0.1

if __name__ == "__main__":
    num_episodes = 1000
    for episode in range(num_episodes):
        game.reset()
        total_rewards = 0
        for step in range(100):

            state = agent.get_state(game)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            if np.random.uniform(0, 1) < epsilon:
                action = agent.predict_action()
            else:
                action = torch.argmax(model(state)).item()

            game_over, reward, score = game.step(action)
            new_state = agent.get_state(game)
            new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
            total_rewards += reward
            target = reward
            state = new_state

            if game_over:
                break

        print("Reward:", total_rewards)
