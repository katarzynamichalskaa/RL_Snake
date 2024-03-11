import pygame
import time
import random

class Snake():
    def __init__(self, width, height):
        #game properties
        pygame.init()
        self.dis = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        #snake properties
        self.x = width/2
        self.y = height/2
        self.x2 = 0
        self.y2 = 0
        self.speed = 15
        self.unit_per_movement = 5
        self.snake_segments = [(self.x, self.y)]
        #food properties
        self.foodx = self.random(self.width)
        self.foody = self.random(self.height)

    def update(self):
        game_over = False
        #game loop
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #quit
                    game_over = True
                if event.type == pygame.KEYDOWN:
                    #control snake TODO: modifying the way snake moves so that AI can control it
                    self.move(event)
            if self.check_boundaries() or self.check_collision():
                #check if snake hits the bounds or yourself TODO: AI should get negative reward
                game_over = True
            if self.eat():
                #expand snake if he ate a rectangle TODO: AI should get positive reward
                self.expand_snake()
            #TODO: if snake (AI) hits itself should get negative reward
            #TODO:score
            #update snake's pos
            self.update_snake_pos()
            self.x += self.x2
            self.y += self.y2

            self.render()
            self.clock.tick(self.speed)
        self.gameover()
        self.quit()

    def quit(self):
        pygame.quit()
        quit()

    def check_boundaries(self):
        if self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0:
            return True

    def check_collision(self):
        current_snake_pos = (self.x, self.y)
        if current_snake_pos == any(self.snake_segments):
            return True

    def gameover(self):
        self.message("Game over", (255, 91, 165))
        pygame.display.update()
        time.sleep(2)

    def message(self, msg, color):
        font_style = pygame.font.SysFont('comicsansms', 30)
        mesg = font_style.render(msg, True, color)
        self.dis.blit(mesg, [self.width / 2, self.height / 2])

    def render(self):
        self.dis.fill((234, 195, 184))
        #snake render
        for segment in self.snake_segments:
            pygame.draw.rect(self.dis, (255, 91, 165),[segment[0], segment[1], self.unit_per_movement, self.unit_per_movement])
        #food render
        pygame.draw.rect(self.dis, (255, 0, 255), [self.foodx, self.foody, self.unit_per_movement, self.unit_per_movement])
        pygame.display.update()

    def update_snake_pos(self):
        for i in range(len(self.snake_segments) - 1, 0, -1):
            self.snake_segments[i] = self.snake_segments[i - 1]
        self.snake_segments[0] = (self.x, self.y)

    def eat(self):
        if self.x == self.foodx and self.y == self.foody:
            self.foodx = self.random(self.width)
            self.foody = self.random(self.height)
            return True

    def expand_snake(self):
        self.snake_segments.append((self.x, self.y))

    def random(self, dim):
        cord = round(random.randrange(0, dim - self.unit_per_movement) / 10.0) * 10.0
        return cord

    def move(self, event):
        if event.key == pygame.K_LEFT:
            self.x2 = -self.unit_per_movement
            self.y2 = 0
        elif event.key == pygame.K_RIGHT:
            self.x2 = self.unit_per_movement
            self.y2 = 0
        elif event.key == pygame.K_UP:
            self.y2 = -self.unit_per_movement
            self.x2 = 0
        elif event.key == pygame.K_DOWN:
            last_key = event.key
            self.y2 = self.unit_per_movement
            self.x2 = 0




