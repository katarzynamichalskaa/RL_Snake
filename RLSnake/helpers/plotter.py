from IPython import display
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        plt.ion()

    def plot(self, scores, avg_scores):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        # plot
        plt.plot(scores, color='green', label='Score')
        plt.plot(avg_scores, color='red', linestyle='dotted', label='Average Score')

        # label plot
        self.label(scores, avg_scores)

    @staticmethod
    def label(scores, avg_scores):

        plt.legend()
        plt.title(f'Agent is playing Snake')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.ylim(ymin=0)
        plt.grid(True)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(avg_scores) - 1, avg_scores[-1], str(avg_scores[-1]))
        plt.show(block=False)
        plt.pause(.01)
