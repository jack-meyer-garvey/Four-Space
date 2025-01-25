from tkinter import *
import numpy as np


def createWindow(x: int, y: int):
    r = Tk()
    r.geometry(f"{x}x{y}")
    r.configure(background='black')
    r.title("Four Space")
    c = Canvas(r, width=x, height=y, bg='black')
    c.focus_set()
    c.pack()
    c.focus_set()
    return r, c


def xCoordinateToScreen(x: float):
    return resize * (180 + x)


def yCoordinateToScreen(y: float):
    return resize * (126 - y)


def rotateXY(x, y, z, w, angle):
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle), z, w

def rotateXZ(x, y, z, w, angle):
    return x * np.cos(angle) - z * np.sin(angle), y, x * np.sin(angle) + z * np.cos(angle), w

def rotateXW(x, y, z, w, angle):
    return x * np.cos(angle) - w * np.sin(angle), y, z, x*np.sin(angle) + w*np.cos(angle)

def rotateYZ(x, y, z, w, angle):
    return x, y * np.cos(angle) - z * np.sin(angle), y * np.sin(angle) + z * np.cos(angle), w

def rotateYW(x, y, z, w, angle):
    return x, y * np.cos(angle) - w * np.sin(angle), z, y*np.sin(angle) + w*np.cos(angle)

def rotateZW(x, y, z, w, angle):
    return x, y, z * np.cos(angle) - w * np.sin(angle), z * np.sin(angle) + w * np.cos(angle)


class overlay:
    def __init__(self):
        # memory for grid system
        self.sideX = 0
        self.sideY = 0
        # initialize indices
        self.leftGridIndex = None
        self.rightGridIndex = None
        self.textboxIndex = None
        self.textboxText = None
        self.drawRightBox()
        self.drawLeftBox()
        self.drawTextbox()

    def drawRightBox(self):
        self.rightGridIndex = canvas.create_rectangle(338 * resize, 12 * resize, 566 * resize, 240 * resize,
                                                      outline='white', width=4 * resize)

    def drawLeftBox(self):
        self.leftGridIndex = canvas.create_rectangle(34 * resize, 12 * resize, 326 * resize, 240 * resize,
                                                     outline='white', width=4 * resize)

    def drawTextbox(self):
        self.textboxIndex = canvas.create_rectangle(34 * resize, 252 * resize, 566 * resize, 388 * resize,
                                                    outline='white', width=4 * resize)
        self.textboxText = canvas.create_text(46 * resize, 278 * resize, text='', fill="white", anchor=NW,
                                              font=('System', int(12 * resize)), width=508 * resize)

resize = 2
root, canvas = createWindow(int(600 * resize), int(400 * resize))
main = overlay()

square = [-20, -20, -20, 20, 20, 20, 20, -20, -20, -20]
for i in range(len(square)//2):
    a, b = rotateZW(square[i*2], square[i*2+1], 0, 0, np.pi*3/4)[0:2]
    square[i*2] = xCoordinateToScreen(a)
    square[i*2+1] = yCoordinateToScreen(b)
thing = canvas.create_line(*square, fill='white', width=4*resize, joinstyle='miter')

mainloop()
