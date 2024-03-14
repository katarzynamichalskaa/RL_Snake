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

    def update(self):
        game_over = False
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                if event.type == pygame.KEYDOWN:
                    self.snake.move(event) # control snake TODO: modifying the way snake moves so that AI can control it

            if self.snake.check_boundaries() or self.snake.check_collision() or self.snake.wall_detection(self.snake.x,
                                                                                                          self.snake.y):
                game_over = True  # check if snake hits the bounds or yourself TODO: AI should get negative reward

            if self.snake.eat():
                self.snake.expand_snake() # expand snake if he ate a rectangle TODO: AI should get positive reward

            self.snake.x += self.snake.x2
            self.snake.y += self.snake.y2

            self.snake.update_snake_pos()
            self.snake.render()
            self.score()
            self.snake.wall_detection(self.snake.x, self.snake.y)
            self.clock.tick(self.snake.speed)
            pygame.display.update()

        self.gameover()
        self.quit()

    def gameover(self):
        self.send_message("Game over", (0, 255, 0), [self.width / 2, self.height / 2], 30)
        time.sleep(2)

    def score(self):
        self.send_message("Score: " + str(len(self.snake.snake_segments)-1), (0, 255, 0), [self.width / 50, self.height / 50], 30)

    def send_message(self, msg, color, dest, size):
        font_style = pygame.font.SysFont('comicsansms', size)
        mesg = font_style.render(msg, True, color)
        self.dis.blit(mesg, dest)
        pygame.display.update()

    def quit(self):
        pygame.quit()
        quit()
