import pygame
import random


class Snake:
    def __init__(self, width, height, display_surface):
        self.dis = display_surface
        self.width = width
        self.height = height

        # snake properties
        self.x = width / 2
        self.y = height / 2
        self.x2 = 0
        self.y2 = 0
        self.direction = 'N'
        self.speed = 15
        self.unit_per_movement = 5
        self.snake_segments = [(self.x, self.y)]

        self.wall_width = 75
        self.wall_length = 155
        self.number_of_walls = 5
        self.walls_position = self.create_walls()

        # while snake's pos in walls, recreate walls
        while self.wall_detection(self.x, self.y):
            self.walls_position = self.create_walls()

        self.foodx, self.foody = self.random([self.width, self.height])

        # while foods' pos in walls, recreate foods' pos
        while self.wall_detection(self.foodx, self.foody):
            self.foodx, self.foody = self.random([self.width, self.height])

    def update_snake_pos(self):
        for i in range(len(self.snake_segments) - 1, 0, -1):
            self.snake_segments[i] = self.snake_segments[i - 1]
        self.snake_segments[0] = (self.x, self.y)

    def create_walls(self):
        walls_position = []
        for i in range(0, self.number_of_walls):
            x, y = self.random([self.width, self.height])
            position = (x, y, self.wall_width, self.wall_length)
            walls_position.append(position)
        return walls_position

    def wall_detection(self, x, y):
        walls_area = self.walls_position
        for wall in walls_area:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            point_rect = pygame.Rect(x, y, 2.5, 2.5)
            if wall_rect.colliderect(point_rect):
                return True

    def check_boundaries(self):
        if self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0:
            return True

    def check_collision(self):
        segments = self.snake_segments[1:]
        for segment in segments:
            if self.x == float(segment[0]) and self.y == float(segment[1]):
                return True

    def eat(self):
        if self.x == self.foodx and self.y == self.foody:
            x, y = self.random([self.width, self.height])
            while self.wall_detection(x, y) is True:
                x, y = self.random([self.width, self.height])
            self.foodx = x
            self.foody = y
            return True

    def expand_snake(self):
        self.snake_segments.append((self.x, self.y))

    def random(self, dims):
        for dim in dims:
            cord = round(random.randrange(0, dim - self.unit_per_movement) / 10.0) * 10.0
            yield cord

    def move(self, event):
        if (event.key == pygame.K_LEFT) & (self.direction != 'L') & (self.direction != 'R'):
            self.x2 = -self.unit_per_movement
            self.y2 = 0
            self.direction = 'L'
        elif (event.key == pygame.K_RIGHT) & (self.direction != 'R') & (self.direction != 'L'):
            self.x2 = self.unit_per_movement
            self.y2 = 0
            self.direction = 'R'
        elif (event.key == pygame.K_UP) & (self.direction != 'U') & (self.direction != 'D'):
            self.y2 = -self.unit_per_movement
            self.x2 = 0
            self.direction = 'U'
        elif (event.key == pygame.K_DOWN) & (self.direction != 'D') & (self.direction != 'U'):
            self.y2 = self.unit_per_movement
            self.x2 = 0
            self.direction = 'D'

    def render_snake(self):
        for segment in self.snake_segments:
            pygame.draw.rect(self.dis, (255, 0, 0),
                             [segment[0], segment[1], self.unit_per_movement, self.unit_per_movement])

    def render_food(self):
        pygame.draw.rect(self.dis, (0, 0, 255),
                         [self.foodx, self.foody, self.unit_per_movement, self.unit_per_movement])

    def render_walls(self):
        walls_position = self.walls_position
        for single_wall in walls_position:
            pygame.draw.rect(self.dis, (0, 0, 0), [single_wall[0], single_wall[1], self.wall_width, self.wall_length])

    def render(self):
        self.dis.fill((245, 245, 190))
        self.render_snake()
        self.render_food()
        self.render_walls()
