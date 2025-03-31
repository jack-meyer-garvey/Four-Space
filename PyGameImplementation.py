import pygame
import numpy as np

resize = 1
size = (int(600 * resize), int(400 * resize))
pygame.init()
pygame.display.set_caption("Four Space")
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
running = True
pygame.display.set_icon(pygame.image.load('FourSpaceLogo.png'))

hoi = np.zeros(size)
hoi2 = np.zeros(size)+16711680

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            hoi, hoi2 = hoi2, hoi

    pygame.surfarray.blit_array(screen, hoi)
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 60
pygame.quit()