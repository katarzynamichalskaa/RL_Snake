import pygame
import random
from enum import Enum


def random_coords(dims, unit):
    return tuple(round(random.randrange(0, dim - unit) / 10.0) * 10.0 for dim in dims)


class Directions(Enum):
    UP = 'U'
    DOWN = 'D'
    RIGHT = 'R'
    LEFT = 'L'


class Snake:
    def __init__(self, width, height, display_surface):
        self.dis = display_surface
        self.width = width
        self.height = height

        # snake properties
        self.speed = 15
        self.unit_per_movement = 5

        # walls properties
        self.wall_width = 75
        self.wall_length = 155
        self.number_of_walls = 5
        self.reset()

    def reset(self):
        self.x = self.width / 2
        self.y = self.height / 2
        self.x2 = 0
        self.y2 = 0
        self.direction = Directions.UP
        self.snake_segments = [(self.x, self.y)]
        self.create_walls()
        self.create_food()

    def update_snake_pos(self):
        self.snake_segments = [(self.x, self.y)] + self.snake_segments[:-1]

    def create_food(self):
        self.foodx, self.foody = random_coords([self.width, self.height], self.unit_per_movement)
        while self.wall_detection(self.foodx, self.foody):
            self.foodx, self.foody = random_coords([self.width, self.height], self.unit_per_movement)

    def create_walls(self):
        self.walls_position = []
        for _ in range(self.number_of_walls):
            while True:
                x, y = random_coords([self.width, self.height], self.unit_per_movement)
                position = (x, y, self.wall_width, self.wall_length)
                wall_rect = pygame.Rect(position[0], position[1], position[2], position[3])
                if not any(wall_rect.colliderect(pygame.Rect(w[0], w[1], w[2], w[3])) for w in self.walls_position):
                    if not self.wall_detection(x, y):
                        self.walls_position.append(position)
                        break

    def wall_detection(self, x, y):
        walls_area = self.walls_position
        for wall in walls_area:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            point_rect = pygame.Rect(x, y, 2.5, 2.5)
            if wall_rect.colliderect(point_rect):
                return True

    def check_boundaries(self):
        return self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0

    def check_collision(self):
        return any((self.x, self.y) == segment for segment in self.snake_segments[1:])

    def eat(self):
        if (self.x, self.y) == (self.foodx, self.foody):
            self.create_food()
            return True

    def expand_snake(self):
        self.snake_segments.append((self.x, self.y))

    def move(self, event):
        if event.key == pygame.K_LEFT and self.direction != Directions.LEFT and self.direction != Directions.RIGHT:
            self.x2 = -self.unit_per_movement
            self.y2 = 0
            self.direction = Directions.LEFT
        elif event.key == pygame.K_RIGHT and self.direction != Directions.RIGHT and self.direction != Directions.LEFT:
            self.x2 = self.unit_per_movement
            self.y2 = 0
            self.direction = Directions.RIGHT
        elif event.key == pygame.K_UP and self.direction != Directions.UP and self.direction != Directions.DOWN:
            self.y2 = -self.unit_per_movement
            self.x2 = 0
            self.direction = Directions.UP
        elif event.key == pygame.K_DOWN and self.direction != Directions.DOWN and self.direction != Directions.UP:
            self.y2 = self.unit_per_movement
            self.x2 = 0
            self.direction = Directions.DOWN
