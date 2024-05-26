from agent import Agent
from game import Game
import numpy as np
import torch
from plotter import Plotter

MODEL_LOADING_BOOL = True

game = Game(500, 500)
plotter = Plotter()
agent = Agent()
alpha = 0.5
epsilon = 3000
plot_scores = []
avg_scores = []

if MODEL_LOADING_BOOL:
    model = Agent().load_model()
    agent.model = model

if __name__ == "__main__":
    num_episodes = 15000
    steps_per_episode = 50000

    for episode in range(num_episodes):

        # reset game every episode
        game.reset()

        for step in range(steps_per_episode):
            # get state
            state = agent.get_state(game)

            # exploration vs exploitation
            if not MODEL_LOADING_BOOL:
                epsilon = epsilon - 1
            else:
                epsilon = -1

            # choose random action
            if np.random.randint(0, 3000) < epsilon:
                action = agent.random_action()
            # choose best action based on Q value
            else:
                action = [0, 0, 0, 0]
                state_torch = torch.tensor(state, dtype=torch.float)
                index_action = torch.argmax(agent.model(state_torch)).item()
                action[index_action] = 1

            # take step
            game_over, reward, score = game.step(action)

            # get new state based on taken step
            new_state = agent.get_state(game)

            # train model
            agent.trainer.train_model(state, action, reward, new_state, game_over)

            # remember
            agent.remember(state, action, reward, new_state, game_over)

            if game_over or step == steps_per_episode - 1:
                agent.number_of_games += 1

                # train model from random sample
                agent.train_long_memory()

                # append scores
                plot_scores.append(score)
                avg_score = sum(plot_scores) / len(plot_scores)
                avg_scores.append(avg_score)

                # plot every 50 games
                if agent.number_of_games % 100 == 0:
                    plotter.plot(plot_scores, avg_scores)

                print('Game', agent.number_of_games, 'Score', score)

                if agent.number_of_games % 1000 == 0:
                    agent.save_model()

                break





