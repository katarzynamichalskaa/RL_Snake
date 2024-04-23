from agent import Agent
from game import Game
import numpy as np
import torch

game = Game(100, 100)
agent = Agent()
alpha = 0.5
epsilon = 80
plot_scores = []
plot_mean_scores = []

if __name__ == "__main__":
    num_episodes = 10000
    for episode in range(num_episodes):
        game.reset()
        total_rewards = 0

        for step in range(200):

            state = agent.get_state(game)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            epsilon = epsilon - agent.number_of_games

            if np.random.randint(0, 100) < epsilon:
                action = agent.predict_action()
            else:
                action = [0, 0, 0]
                index_action = torch.argmax(agent.model(state)).item()
                action[index_action] = 1

            game_over, reward, score = game.step(action)
            new_state = agent.get_state(game)
            new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)

            #agent.trainer.train_model(state, action, reward, new_state, game_over)

            agent.remember(state, action, reward, new_state, game_over)

            total_rewards += reward
            state = new_state

            if game_over or step == 199:
                agent.number_of_games += 1
                agent.train_long_memory()
                print("Reward:", total_rewards)

                break

