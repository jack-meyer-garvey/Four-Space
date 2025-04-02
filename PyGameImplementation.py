import pygame
import numpy as np

resize = 2
size = (int(600 * resize), int(400 * resize))
pygame.init()
pygame.display.set_caption("Four Space")
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
running = True
pygame.display.set_icon(pygame.image.load('FourSpaceLogo.png'))

# hoi is the main canvas. This gets drawn to screen every frame.
hoi = np.zeros(size)

class pixel_grid:
    binary_font = {' ': 0, 'A': 75247420147096911872, 'B': 555871685767213285376, 'C': 260722531473394434048,
                   'D': 555871571418003996672, 'E': 574173188683679629312, 'F': 574173188683679137792,
                   'G': 260722531680089735168, 'H': 316064021605081710592, 'I': 572430066098342166528,
                   'J': 129272458469370691584, 'K': 316211488173855375360, 'L': 297471904432734044160,
                   'M': 316075192918097821696, 'N': 316072940980845182976, 'O': 260723666238650646528,
                   'P': 555871572310815670272, 'Q': 260723666238650647296, 'R': 555871572311893639168,
                   'S': 260722512842363174912, 'T': 572430066098341281792, 'U': 316063898459779301376,
                   'V': 316063898455991517184, 'W': 316063933915241709568, 'X': 316055902330185875456,
                   'Y': 316063835986740772864, 'Z': 571995469015671603200, 'a': 15772434787893248,
                   'b': 297478739066306101248, 'c': 15913240450498560, 'd': 18605664263779811328,
                   'e': 15914271242649600, 'f': 130315103840515325952, 'g': 14805001994404124,
                   'h': 297478739066272382976, 'i': 73818536953645072384, 'j': 36909268476822098456,
                   'k': 297473048206155218944, 'l': 517089833877213511680, 'm': 29459570003443712,
                   'n': 24990877705863168, 'o': 15913309706846208, 'p': 24990877739585568,
                   'q': 14805001994404098, 'r': 30505404686336000, 's': 15913094958481408,
                   't': 148761847914228285440, 'u': 19291009435729920, 'v': 19291005639589888,
                   'w': 23830070468902912, 'x': 19228539665022976, 'y': 19291005639590960,
                   'z': 34920768539164672, '.': 16908288, ',': 16910336, '?': 260705518998341681152,
                   '!': 74367976108166610944, '(':4797315458791173652992, ')': 18963543407680091719680,
                   "'": 74367940646769000448}


    def __init__(self, screen_size, pixel_size, main_canvas, background_color=707322):
        self.screen_size = screen_size
        self.pixel_size = pixel_size
        self.gridX = screen_size[0] // pixel_size
        self.gridY = screen_size[1] // pixel_size
        self.main_canvas = main_canvas
        self.background_color = background_color
        self.layers = []
        self.add_blank_layer()

    def add_blank_layer(self):
        self.layers.append((np.zeros(self.screen_size), np.ones(self.screen_size)))

    def apply_layers(self):
        self.main_canvas *= 0
        self.main_canvas += self.background_color
        for color, transparency in self.layers:
            self.main_canvas *= transparency
            self.main_canvas += color

    def draw(self, pixelated_array, transparency_array, x_position, y_position, layer=0):
        draw_slice = self.layers[layer][0][x_position*self.pixel_size:x_position*self.pixel_size+pixelated_array.shape[0],
                     y_position*self.pixel_size:y_position*self.pixel_size+pixelated_array.shape[1]]
        draw_slice *= transparency_array
        draw_slice += pixelated_array
        transparency_slice = self.layers[layer][1][x_position*self.pixel_size:x_position*self.pixel_size+pixelated_array.shape[0],
                     y_position*self.pixel_size:y_position*self.pixel_size+pixelated_array.shape[1]]
        transparency_slice *= transparency_array

    def draw_binary(self, num, x_position, y_position, color=2**24-1, layer=0, x_width=7, y_width=12):
        A = np.zeros((x_width*self.pixel_size, y_width*self.pixel_size))
        B = np.array(np.array_split(np.array([num >> i & 1 for i in range(x_width*y_width-1, -1, -1)]), y_width)).transpose()
        Y = A.shape[0]
        X = A.shape[1]
        k = self.pixel_size
        for y in range(0, k):
            for x in range(0, k):
                A[y:Y:k, x:X:k] = B
        self.draw(A*color, 1-A, x_position, y_position, layer)

    def draw_text(self, text, x_position, y_position, layer=-1, color=2**24-1):
        for i, _ in enumerate(text):
            self.draw_binary(pixel_grid.binary_font[_], x_position+7*i, y_position, layer=layer, color=color)


n = 4
canvas = pixel_grid(size, n, hoi)

canvas.draw_text("What the fuck? Fuckin' hell!", 4, 10)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            pass

    canvas.apply_layers()
    pygame.surfarray.blit_array(screen, hoi)
    pygame.display.flip()
    clock.tick(30)  # limits FPS to 60
    if clock.get_fps() < 28 and clock.get_fps():
        print(clock.get_fps())
pygame.quit()