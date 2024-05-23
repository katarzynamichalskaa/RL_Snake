import time

from agent import Agent
from game import Game
import numpy as np
import torch
from plotter import Plotter

game = Game(500, 500)
plotter = Plotter()
# new agent
agent = Agent()


if True:
    model = Agent().load_model()
    load_status='loaded'

agent.model = model


alpha = 0.5
epsilon = 80
plot_scores = []
avg_scores = []

if __name__ == "__main__":
    num_episodes = 500
    steps_per_episode = 100000

    for episode in range(num_episodes):

        # reset game every episode
        game.reset()

        for step in range(steps_per_episode):
            # get state
            state = agent.get_state(game)

            # exploration vs exploitation
            if agent.number_of_games < 400:

                if epsilon > 10 and load_status != 'loaded':
                    epsilon = epsilon - agent.number_of_games
                else:
                    epsilon = 0
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

                # train model from random sample
                agent.train_long_memory()

                # append scores
                plot_scores.append(score)
                avg_score = sum(plot_scores) / len(plot_scores)
                avg_scores.append(avg_score)

                # plot every 50 games
                if agent.number_of_games % 10 == 0:
                    plotter.plot(plot_scores, avg_scores)
                if agent.number_of_games > 800:
                    game.snake.speed = 30



                print('Game', agent.number_of_games, 'Score', score)

                break
    #agent.save_model()
    #time.sleep(10)
    #agent.load_model()



