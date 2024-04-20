import pygame
import random
from enum import Enum
import numpy as np


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
        self.reset_snake()

    def reset_snake(self):
        self.x = self.width / 2
        self.y = self.height / 2
        self.x2 = 0
        self.y2 = 0
        self.direction = Directions.DOWN
        self.snake_segments = [(self.x, self.y)]
        #self.create_walls()
        self.create_food()

    def update_snake_pos(self):
        self.snake_segments = [(self.x, self.y)] + self.snake_segments[:-1]

    def create_food(self):
        self.foodx, self.foody = random_coords([self.width, self.height], self.unit_per_movement)
        while self.foodx in self.snake_segments or self.foody in self.snake_segments: #self.wall_detection(self.foodx, self.foody) or
            self.create_food()

    def create_walls(self):
        self.walls_position = []
        for i in range(0, self.number_of_walls):
            x, y = random_coords([self.width, self.height], self.unit_per_movement)
            position = (x, y, self.wall_width, self.wall_length)
            self.walls_position.append(position)
        while self.wall_detection(self.x, self.y):
            self.walls_position = self.create_walls()
        return self.walls_position

    def wall_detection(self, x, y):
        walls_area = self.walls_position
        for wall in walls_area:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            point_rect = pygame.Rect(x, y, 2.5, 2.5)
            if wall_rect.colliderect(point_rect):
                return True

    def check_boundaries(self):
        return self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0

    def check_danger(self, offset):
        danger_zones = [0, 0, 0, 0]
        if self.x + offset >= self.width:
            danger_zones[2] = 1         # right wall
        elif self.x - offset < 0:
            danger_zones[0] = 1         # left wall
        elif self.y + offset >= self.height:
            danger_zones[3] = 1         # down wall
        elif self.y - offset < 0:
            danger_zones[1] = 1         # up wall
        return danger_zones

    def check_collision(self):
        return any((self.x, self.y) == segment for segment in self.snake_segments[1:])

    def eat(self):
        if (self.x, self.y) == (self.foodx, self.foody):
            self.create_food()
            return True

    def expand_snake(self):
        self.snake_segments.append((self.x, self.y))

    def move(self, action):
        clock_wise = [Directions.RIGHT, Directions.DOWN, Directions.LEFT, Directions.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        if self.direction == Directions.RIGHT:
            self.x += self.unit_per_movement
        elif self.direction == Directions.LEFT:
            self.x -= self.unit_per_movement
        elif self.direction == Directions.DOWN:
            self.y += self.unit_per_movement
        elif self.direction == Directions.UP:
            self.y -= self.unit_per_movement