import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from collections import deque
import numpy as np
import re

class PixelGrid:
    binary_font = {' ': 0, '': 0,
                   'A': 75247420147096911872, 'B': 555871685767213285376, 'C': 260722531473394434048,
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
                   "'": 74367940646769000448, "=": 272695526686720, "<": 292805448346697728,
                   ">": 1157442766230519808, "ge": 148152674077506539264, "le": 37479097388377316864,
                   "0": 260725953772592234496, "1": 77826740622004813824, "2": 260705519000557158400,
                   "3": 260705553975016357888, "4": 56071091979129225216, "5": 574173187644838051840,
                   "6": 260722654687953092608, "7": 571994325244371533824, "8": 260723639850371579904,
                   "9": 260723666092621758464}

    def __init__(self, screen_size, pixel_size, background_color=2736806):
        self.screen_size = screen_size
        self.shift = (0, 0)
        self.pixel_size = pixel_size
        self.grid_size = (screen_size[0] // pixel_size, screen_size[1] // pixel_size)
        self.canvas_size = (self.grid_size[0] * self.pixel_size, self.grid_size[1] * self.pixel_size)
        self.main_canvas = np.zeros(screen_size)
        self.background_color = background_color
        self.layers = []
        self.add_blank_layer()

    def window_resize_shift(self, x, y):
        self.shift = ((x - self.screen_size[0]) // 2, (y - self.screen_size[1]) // 2)

    def add_blank_layer(self):
        self.layers.append((np.zeros(self.grid_size), np.ones(self.grid_size)))

    def erase(self, layer, x1, y1, x2, y2):
        color, transparency = self.layers[layer]
        color[x1:x2, y1:y2] = 0
        transparency[x1:x2, y1:y2] = 1

    def apply_layers(self):
        sum_layers = np.zeros(self.grid_size) + self.background_color
        for color, transparency in self.layers:
            sum_layers *= transparency
            sum_layers += color
        Y = self.canvas_size[0]
        X = self.canvas_size[1]
        k = self.pixel_size
        for y in range(0, k):
            for x in range(0, k):
                self.main_canvas[y:Y:k, x:X:k] = sum_layers

    def draw(self, pixelated_array, transparency_array, x_position, y_position, layer=0):
        if x_position < 0:
            pixelated_array = pixelated_array[-x_position:]
            transparency_array = transparency_array[-x_position:]
            x_position = 0
        if y_position < 0:
            pixelated_array = pixelated_array[:, -y_position:]
            transparency_array = transparency_array[:, -y_position:]
            y_position = 0
        if x_position + pixelated_array.shape[0] > self.grid_size[0]:
            pixelated_array = pixelated_array[:self.grid_size[0]-(x_position+pixelated_array.shape[0])]
            transparency_array = transparency_array[:self.grid_size[0]-(x_position+transparency_array.shape[0])]
        if y_position + pixelated_array.shape[1] > self.grid_size[1]:
            pixelated_array = pixelated_array[:,:self.grid_size[1]-(y_position+pixelated_array.shape[1])]
            transparency_array = transparency_array[:,:self.grid_size[1]-(y_position+transparency_array.shape[1])]
        x_width = pixelated_array.shape[0]
        y_width = pixelated_array.shape[1]
        draw_slice = self.layers[layer][0][x_position:x_position+x_width, y_position:y_position+y_width]
        draw_slice *= transparency_array
        draw_slice += pixelated_array
        transparency_slice = self.layers[layer][1][x_position:x_position+x_width, y_position:y_position+y_width]
        transparency_slice *= transparency_array

    def draw_binary(self, num, x_position, y_position, color=2**24-1, background_color=None, layer=0, x_width=7, y_width=12):
        B = np.array(
            np.array_split(np.array([num >> i & 1 for i in range(x_width * y_width - 1, -1, -1)]), y_width)).transpose()
        if background_color is None:
            self.draw(B*color, 1-B, x_position, y_position, layer=layer)
        else:
            self.draw(B*color+(1-B)*background_color, np.zeros((x_width, y_width)), x_position, y_position, layer=layer)

    def draw_string(self, text, x_position, y_position, layer=-1, color=2**24-1, background_color=None):
        letter = x_position
        i = 0
        while i<len(text):
            if text[i] == '<':
                command = text[i+1:text.find('>', i+1)]
                self.draw_binary(PixelGrid.binary_font[command], letter, y_position, layer=layer, color=color,
                             background_color=background_color)
                i += len(command) + 1
            else:
                self.draw_binary(PixelGrid.binary_font[text[i]], letter, y_position, layer=layer, color=color,
                                 background_color=background_color)
            letter += 7
            i += 1
        return letter, y_position + 12

    def draw_string_with_indexing(self, text, x_position, y_position, layer=-1, color=2**24-1, background_color=None):
        letter = x_position
        indexing = []
        i = 0
        while i < len(text):
            indexing.append(i)
            if text[i] == '<':
                command = text[i + 1:text.find('>', i + 1)]
                self.draw_binary(PixelGrid.binary_font[command], letter, y_position, layer=layer, color=color,
                                 background_color=background_color)
                i += len(command) + 1
            else:
                self.draw_binary(PixelGrid.binary_font[text[i]], letter, y_position, layer=layer, color=color,
                                 background_color=background_color)
            letter += 7
            i += 1
        return letter, y_position + 12, indexing


# String Rewrite Rule ----------
class Ground:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def no_check_replace(self, text, index):
        return text[:index] + self.right + text[index+len(self.left):]

    def text_index_to_match_index(self, text):
        text_index_to_match_index = {}
        matches = reversed(list(re.finditer(f'(?={self.left})', text)))
        last_match_index = len(text)
        for _ in matches:
            _ = _.start()
            for i in range(_, min(_+len(self.left), last_match_index)):
                text_index_to_match_index[i] = _
            last_match_index = _
        return text_index_to_match_index


# Puzzle ----------
class Puzzle:
    instances = []
    def __init__(self, pGrid, start, end, rules, x_position, y_position, layer=-1, color=2**24-1, background_color=None,
                 highlight_color=12734249):
        Puzzle.instances.append(self)
        self.canvas = pGrid
        self.x_position = x_position
        self.y_position = y_position
        self.path = deque([start])
        self.end = end
        self.current_rule = rules[0]
        self.rules = rules
        self.layer = layer
        self.color = color
        self.background_color = background_color
        self.highlight_color = highlight_color
        self.x_position_end, self.y_position_end, self.indexing = pGrid.draw_string_with_indexing(start, x_position,
                                        y_position, layer=layer, color=color, background_color=background_color)
        self.is_hovering = False
        self.matched_index = None
        self.text_index_to_match_index = self.current_rule.text_index_to_match_index(start)

    def pixel_to_text_index(self, x_pixel):
        return self.indexing[(x_pixel - self.x_position)//7]

    def is_within(self, x, y):
        return (self.x_position <= x < self.x_position_end
            and self.y_position <= y < self.y_position_end)

    def mouse_match_select(self, pos):
        x, y = (pos[0] - self.canvas.shift[0]) // self.canvas.pixel_size, (pos[1] - self.canvas.shift[1]) // self.canvas.pixel_size
        if self.is_within(x, y):
            _ = self.pixel_to_text_index(x)
            if _ in self.text_index_to_match_index:
                self.is_hovering = True
                self.matched_index = self.text_index_to_match_index[_]
                self.highlight_match(self.text_index_to_match_index[_])
            elif self.is_hovering:
                self.is_hovering = False
                self.un_highlight()
        elif self.is_hovering:
            self.is_hovering = False
            self.un_highlight()

    def highlight_match(self, index):
        self.canvas.erase(self.layer, self.x_position, self.y_position, self.x_position_end, self.y_position_end)
        hl_start = index
        hl_end = hl_start + len(self.current_rule.left)
        x_end = self.canvas.draw_string(self.path[-1][:hl_start],
                                               self.x_position, self.y_position, layer=self.layer, color=self.color,
                                               background_color=self.background_color)[0]
        x_end = self.canvas.draw_string(self.path[-1][hl_start:hl_end], x_end,
                                               self.y_position, layer=self.layer, color=self.color,
                                               background_color=self.highlight_color)[0]
        self.canvas.draw_string(self.path[-1][hl_end:], x_end,
                                self.y_position, layer=self.layer, color=self.color,
                                background_color=self.background_color)

    def un_highlight(self):
        self.canvas.erase(self.layer, self.x_position, self.y_position, self.x_position_end, self.y_position_end)
        self.canvas.draw_string(self.path[-1], self.x_position, self.y_position, layer=self.layer, color=self.color,
                               background_color=self.background_color)

    def apply_rule(self, pos):
        if self.is_hovering:
            new_node = self.current_rule.no_check_replace(self.path[-1], self.matched_index)
            self.path.append(new_node)
            self.canvas.erase(self.layer, self.x_position, self.y_position, self.x_position_end, self.y_position_end)
            self.x_position_end, self.y_position_end = self.canvas.draw_string(new_node, self.x_position, self.y_position, layer=self.layer,
                                                                         color=self.color, background_color=self.background_color)
            self.text_index_to_match_index = self.current_rule.text_index_to_match_index(new_node)
            self.mouse_match_select(pos)




# Main -------------------------------
resize = 2
n = 3 * resize
size = (int(600 * resize), int(360 * resize))

pygame.init()
pygame.display.set_caption("Four Space")
screen = pygame.display.set_mode(size, pygame.RESIZABLE)
game_region = pygame.Surface(size)
clock = pygame.time.Clock()
running = True
pygame.display.set_icon(pygame.image.load('FourSpaceLogo.png'))
canvas = PixelGrid(size, n)

# hoi -------------------------------
R = Ground('aaa', 'b')
boop = Puzzle(canvas, 'aaaaaaaaaaaa', 'aabb', [R], 3, 12, background_color=0)

# Pygame Loop -----------------------------------------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            boop.mouse_match_select(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                boop.apply_rule(event.pos)
        elif event.type == pygame.WINDOWRESIZED:
            canvas.window_resize_shift(event.x, event.y)

    canvas.apply_layers()
    pygame.surfarray.blit_array(game_region, canvas.main_canvas)
    screen.blit(game_region, canvas.shift)
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60
    #if clock.get_fps() < 58 and clock.get_fps():
    #    print(clock.get_fps())

pygame.quit()