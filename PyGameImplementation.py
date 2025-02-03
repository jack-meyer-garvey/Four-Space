import pygame
import numpy as np

def tesseractPoints(x=1, y=1, z=1, w=1):
    return list(np.array([[[-x, -y, -z, -w, 1], [-x, -y, -z, w, 1], [-x, -y, z, w, 1], [-x, -y, z, -w, 1]],
                [[-x, y, -z, -w, 1], [-x, y, -z, w, 1], [-x, y, z, w, 1], [-x, y, z, -w, 1]],
                [[x, -y, -z, -w, 1], [x, -y, -z, w, 1], [x, -y, z, w, 1], [x, -y, z, -w, 1]],
                [[x, y, -z, -w, 1], [x, y, -z, w, 1], [x, y, z, w, 1], [x, y, z, -w, 1]],
                [[-x, -y, -z, -w, 1], [-x, -y, -z, w, 1], [-x, y, -z, w, 1], [-x, y, -z, -w, 1]],
                [[-x, -y, z, -w, 1], [-x, -y, z, w, 1], [-x, y, z, w, 1], [-x, y, z, -w, 1]],
                [[x, -y, -z, -w, 1], [x, -y, -z, w, 1], [x, y, -z, w, 1], [x, y, -z, -w, 1]],
                [[x, -y, z, -w, 1], [x, -y, z, w, 1], [x, y, z, w, 1], [x, y, z, -w, 1]],
                [[-x, -y, -z, -w, 1], [-x, -y, z, -w, 1], [-x, y, z, -w, 1], [-x, y, -z, -w, 1]],
                [[-x, -y, -z, w, 1], [-x, -y, z, w, 1], [-x, y, z, w, 1], [-x, y, -z, w, 1]],
                [[x, -y, -z, -w, 1], [x, -y, z, -w, 1], [x, y, z, -w, 1], [x, y, -z, -w, 1]],
                [[x, -y, -z, w, 1], [x, -y, z, w, 1], [x, y, z, w, 1], [x, y, -z, w, 1]],
                [[-x, -y, -z, -w, 1], [-x, -y, -z, w, 1], [x, -y, -z, w, 1], [x, -y, -z, -w, 1]],
                [[-x, -y, z, -w, 1], [-x, -y, z, w, 1], [x, -y, z, w, 1], [x, -y, z, -w, 1]],
                [[-x, y, -z, -w, 1], [-x, y, -z, w, 1], [x, y, -z, w, 1], [x, y, -z, -w, 1]],
                [[-x, y, z, -w, 1], [-x, y, z, w, 1], [x, y, z, w, 1], [x, y, z, -w, 1]],
                [[-x, -y, -z, -w, 1], [-x, -y, z, -w, 1], [x, -y, z, -w, 1], [x, -y, -z, -w, 1]],
                [[-x, -y, -z, w, 1], [-x, -y, z, w, 1], [x, -y, z, w, 1], [x, -y, -z, w, 1]],
                [[-x, y, -z, -w, 1], [-x, y, z, -w, 1], [x, y, z, -w, 1], [x, y, -z, -w, 1]],
                [[-x, y, -z, w, 1], [-x, y, z, w, 1], [x, y, z, w, 1], [x, y, -z, w, 1]],
                [[-x, -y, -z, -w, 1], [-x, y, -z, -w, 1], [x, y, -z, -w, 1], [x, -y, -z, -w, 1]],
                [[-x, -y, -z, w, 1], [-x, y, -z, w, 1], [x, y, -z, w, 1], [x, -y, -z, w, 1]],
                [[-x, -y, z, -w, 1], [-x, y, z, -w, 1], [x, y, z, -w, 1], [x, -y, z, -w, 1]],
                [[-x, -y, z, w, 1], [-x, y, z, w, 1], [x, y, z, w, 1], [x, -y, z, w, 1]]]))

class fourSpace:
    def __init__(self, xN, yN, zN, wN, sideLength):
        self.xN = xN
        self.yN = yN
        self.zN = zN
        self.wN = wN
        self.sideLength = sideLength

    def drawGrid(self):
        pass


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