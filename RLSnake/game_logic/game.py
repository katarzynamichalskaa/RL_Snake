import pygame
import time
from snake import Snake
from enum import Enum
from utils import DISPLAY_BOOL


class Color(Enum):
    SNAKE = (0, 128, 0)
    FOOD = (255, 0, 0)
    WALL = (0, 0, 0)
    BACKGROUND = (245, 245, 190)


class Game:
    def __init__(self, width, height):
        if DISPLAY_BOOL:
            pygame.init()
            pygame.display.set_caption('Snake')
            self.dis = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.snake = Snake(width, height)
        self.score = len(self.snake.snake_segments) - 1

    def step(self, action):  # modified to step
        game_over = False
        reward = 0

        # events
        if DISPLAY_BOOL:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    self.game_over()
                    game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        game_over = True

        # move the snake
        self.snake.move(action)
        self.snake.x += self.snake.x2
        self.snake.y += self.snake.y2
        self.snake.update_snake_pos()

        # check game over conditions
        if (self.snake.check_boundaries() or self.snake.check_collision(self.snake.snake_segments[1:])
                or self.snake.wall_detection(self.snake.x, self.snake.y)):
            game_over = True
            reward = -12

        # check if the snake eats food
        if self.snake.eat():
            self.score += 1
            self.snake.expand_snake()
            reward = 10

        # render the game
        if DISPLAY_BOOL:
            self.render()
            self.display_score()
            self.clock.tick(self.snake.speed)
            pygame.display.update()

        return game_over, reward, self.score

    def game_over(self):
        self.send_message("Game over", (246, 0, 0), [self.width / 2, self.height / 2], 30)
        time.sleep(2)

    def display_score(self):
        self.send_message("Score: " + str(self.score), (0, 153, 0), [self.width / 50, self.height / 50], 30)

    def send_message(self, msg, color, dest, size):
        font_style = pygame.font.SysFont('font.ttf', size)
        mes = font_style.render(msg, True, color)
        self.dis.blit(mes, dest)
        pygame.display.update()

    @staticmethod
    def quit():
        pygame.quit()

    def render(self):
        self.dis.fill(Color.BACKGROUND.value)

        self.render_objects(self.snake.snake_segments,
                            Color.SNAKE.value,
                            self.snake.unit_per_movement,
                            self.snake.unit_per_movement)
        self.render_objects([(self.snake.food_x, self.snake.food_y)],
                            Color.FOOD.value,
                            self.snake.unit_per_movement,
                            self.snake.unit_per_movement)

        self.render_objects(self.snake.walls_position,
                            Color.WALL.value,
                            self.snake.wall_width,
                            self.snake.wall_length)

    def render_objects(self, objects, color, width, height):
        for obj in objects:
            pygame.draw.rect(self.dis, color, [obj[0], obj[1], width, height])

    def reset(self):
        self.score = 0
        self.snake.reset_snake()
