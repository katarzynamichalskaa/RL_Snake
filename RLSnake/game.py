import pygame
import time
from snake import Snake


class Game:
    def __init__(self, width, height):
        pygame.init()
        self.dis = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.snake = Snake(width, height, self.dis)
        self.score = len(self.snake.snake_segments) - 1

    def step(self):   # modified to step
        game_over = False
        reward = 0
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                if event.type == pygame.KEYDOWN:
                    self.snake.move(event) # control snake TODO: modify the way snake moves so that AI can control it

            if self.snake.check_boundaries() or self.snake.check_collision() or self.snake.wall_detection(self.snake.x,
                                                                                                          self.snake.y):
                game_over = True  # AI should get negative reward
                reward = -10

            if self.snake.eat():
                self.score += 1
                self.snake.expand_snake() # AI get positive reward
                reward = 10

            self.snake.x += self.snake.x2
            self.snake.y += self.snake.y2

            self.snake.update_snake_pos()
            self.snake.render()
            self.display_score()
            self.snake.wall_detection(self.snake.x, self.snake.y)
            self.clock.tick(self.snake.speed)
            pygame.display.update()

        self.gameover()
        self.quit()
        #return game_over, reward, self.score

    def gameover(self):
        self.send_message("Game over", (246, 0, 0), [self.width / 2, self.height / 2], 30)
        time.sleep(2)

    def display_score(self):
        self.send_message("Score: " + str(self.score), (0, 153, 0), [self.width / 50, self.height / 50], 30)

    def send_message(self, msg, color, dest, size):
        font_style = pygame.font.SysFont('font.ttf', size)
        mesg = font_style.render(msg, True, color)
        self.dis.blit(mesg, dest)
        pygame.display.update()

    def quit(self):
        pygame.quit()
        quit()
