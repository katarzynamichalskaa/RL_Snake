from agent import Agent
from game import Game
import numpy as np
import torch
from plotter import Plotter

game = Game(350, 350)
plotter = Plotter()
agent = Agent()
alpha = 0.5
epsilon = 80
plot_scores = []

if __name__ == "__main__":
    num_episodes = 10000
    steps_per_episode = 10000

    for episode in range(num_episodes):

        # reset game every episode
        game.reset()

        for step in range(steps_per_episode):
            # get state
            state = agent.get_state(game)

            # exploration vs exploitation
            if agent.number_of_games < 150:
                if epsilon > 10:
                    epsilon = epsilon - agent.number_of_games
            else:
                epsilon = 0

            # choose random action
            if np.random.randint(0, 200) < epsilon:
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
                # if agent.number_of_games > 150:
                #    game.snake.speed = 5
                # train model from random sample
                agent.train_long_memory()

                # plot
                print('Game', agent.number_of_games, 'Score', score)
                plot_scores.append(score)
                plotter.plot(plot_scores)

                break

