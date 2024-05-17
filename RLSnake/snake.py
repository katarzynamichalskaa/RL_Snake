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
    NONE = 'None'


class Snake:
    def __init__(self, width, height, display_surface):
        self.dis = display_surface
        self.width = width
        self.height = height

        # food properties
        self.food_y = None
        self.food_x = None

        # snake properties
        self.speed = 500
        self.unit_per_movement = 5
        self.y = None
        self.x = None
        self.y2 = None
        self.x2 = None
        self.snake_segments = None
        self.direction = None
        self.first_game = True

        # walls properties
        self.wall_width = 75
        self.wall_length = 155
        self.number_of_walls = 3
        self.create_walls()
        self.reset_snake()

    def reset_snake(self):
        self.x = self.width / 2
        self.y = self.height / 2
        self.x2 = 0
        self.first_game = False
        self.y2 = 0
        self.direction = Directions.NONE
        self.snake_segments = [(self.x, self.y)]
        self.create_walls()
        self.create_food()

    def update_snake_pos(self):
        self.snake_segments = [(self.x, self.y)] + self.snake_segments[:-1]

    def create_food(self):
        self.food_x, self.food_y = random_coords([self.width, self.height], 10)
        while (self.food_x, self.food_y) in self.snake_segments or self.wall_detection(self.food_x, self.food_y):
            self.create_food()

    def create_walls(self):
        self.walls_position = []
        for i in range(0, self.number_of_walls):
            x, y = random_coords([self.width, self.height], self.unit_per_movement)
            position = (x, y, self.wall_width, self.wall_length)
            self.walls_position.append(position)
        if self.first_game:
            self.wall_detection(0, 0)
        else:
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
            danger_zones[2] = 1  # right wall
        elif self.x - offset < 0:
            danger_zones[0] = 1  # left wall
        elif self.y + offset >= self.height:
            danger_zones[3] = 1  # down wall
        elif self.y - offset < 0:
            danger_zones[1] = 1  # up wall
        return danger_zones

    def segment_danger(self, offset):
        danger_zones = [0, 0, 0, 0]
        if len(self.snake_segments) >= 3:
            if any(self.y - offset < segment[1] < self.y and self.x == segment[0] for segment in  # up
                   self.snake_segments[3:]):
                danger_zones[1] = 1
            if any(self.y + offset > segment[1] > self.y and self.x == segment[0] for segment in  # down
                   self.snake_segments[3:]):
                danger_zones[3] = 1
            if any(self.x - offset < segment[0] < self.x and self.y == segment[1] for segment in  # left
                   self.snake_segments[3:]):
                danger_zones[0] = 1
            if any(self.x + offset > segment[0] > self.x and self.y == segment[1] for segment in  # right
                   self.snake_segments[3:]):
                danger_zones[2] = 1
        return danger_zones

    def wall_danger(self, offset):
        danger_zones = [0, 0, 0, 0]
        # LEFT
        if any(self.x - offset < wall[0] + wall[2] and self.x > wall[0] + wall[2] and wall[1] <= self.y <= wall[1] + wall[3] for wall in self.walls_position):
            print("LEFT DANGER")
            danger_zones[0] = 1
        # UP
        if any(self.y - offset < wall[1] + wall[3] and self.y > wall[1] + wall[3] and wall[0] <= self.x <= wall[0]+wall[2] for wall in self.walls_position):
            print("UP DANGER")
            danger_zones[1] = 1
        # RIGHT
        if any(self.x + offset > wall[0] and self.x < wall[0] and wall[1] <= self.y <= wall[1]+wall[3] for wall in self.walls_position):
            print("RIGHT DANGER")
            danger_zones[2] = 1
        # DOWN
        if any(self.y + offset > wall[1] and self.y < wall[1] and wall[0] <= self.x <= wall[0]+wall[2] for wall in self.walls_position):
            print("DOWN DANGER")
            danger_zones[3] = 1

        return danger_zones

    def check_collision(self):
        return any((self.x, self.y) == segment for segment in self.snake_segments[1:])

    def eat(self):
        if (self.x, self.y) == (self.food_x, self.food_y):
            self.create_food()
            return True

    def expand_snake(self):
        self.snake_segments.append((self.x, self.y))

    def move(self, action):
        directions = [Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN]
        opposite_directions = [Directions.RIGHT, Directions.LEFT, Directions.DOWN, Directions.UP]
        actions = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
        movements = [(-self.unit_per_movement, 0), (self.unit_per_movement, 0), (0, -self.unit_per_movement),
                     (0, self.unit_per_movement)]

        for act, dir, mov in zip(actions, directions, movements):
            if np.array_equal(action, act) and self.direction != dir and self.direction != opposite_directions[directions.index(dir)]:
                self.x2, self.y2 = mov
                self.direction = dir
                break
