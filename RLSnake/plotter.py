import matplotlib.pyplot as plt
from IPython import display


class Plotter:
    def __init__(self):
        plt.ion()

    def plot(self, scores):
        if len(scores) == 0:
            avg_score = 0
        else:
            avg_score = sum(scores) / len(scores)

        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title(f'Average score: {avg_score}')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.ylim(ymin=0)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.show(block=False)
        plt.pause(.01)