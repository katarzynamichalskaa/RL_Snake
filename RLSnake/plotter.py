import matplotlib.pyplot as plt
from IPython import display


class Plotter:
    def __init__(self):
        plt.ion()
        self.avg_scores = []

    def plot(self, scores):
        if len(scores) == 0:
            avg_score = 0
        else:
            avg_score = sum(scores) / len(scores)
            self.avg_scores.append(avg_score)

        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title(f'Agent is playing Snake')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, color='green', label='Score')
        plt.plot(self.avg_scores, color='red', linestyle='dotted', label='Average Score')
        plt.legend()
        plt.ylim(ymin=0)
        plt.grid(True)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(self.avg_scores) - 1, self.avg_scores[-1], str(self.avg_scores[-1]))
        plt.show(block=False)
        plt.pause(.01)