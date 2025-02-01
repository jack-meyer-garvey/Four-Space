import pygame
import numpy as np


resize = 1
pygame.init()
pygame.display.set_caption("Four Space")
screen = pygame.display.set_mode((int(600 * resize), int(400 * resize)))
clock = pygame.time.Clock()
running = True
screen.fill("black")
pygame.display.set_icon(pygame.image.load('FourSpaceLogo.png'))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 60
pygame.quit()