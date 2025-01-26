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

def rotationMatrixXY(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle), 0, 0], [np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def rotateXZ(x, y, z, w, angle):
    return x * np.cos(angle) - z * np.sin(angle), y, x * np.sin(angle) + z * np.cos(angle), w

def rotationMatrixXZ(angle):
    return np.matrix([[np.cos(angle), 0, -np.sin(angle), 0], [0, 1, 0, 0], [np.sin(angle), 0, np.cos(angle), 0], [0, 0, 0, 1]])

def rotateXW(x, y, z, w, angle):
    return x * np.cos(angle) - w * np.sin(angle), y, z, x*np.sin(angle) + w*np.cos(angle)

def rotationMatrixXW(angle):
    return np.matrix([[np.cos(angle), 0, 0, -np.sin(angle)], [0, 1, 0, 0], [0, 0, 1, 0], [np.sin(angle), 0, 0, np.cos(angle)]])

def rotateYZ(x, y, z, w, angle):
    return x, y * np.cos(angle) - z * np.sin(angle), y * np.sin(angle) + z * np.cos(angle), w

def rotationMatrixYZ(angle):
    return np.matrix([[1, 0, 0, 0], [0, np.cos(angle), -np.sin(angle), 0], [0, np.sin(angle), np.cos(angle), 0], [0, 0, 0, 1]])

def rotateYW(x, y, z, w, angle):
    return x, y * np.cos(angle) - w * np.sin(angle), z, y*np.sin(angle) + w*np.cos(angle)

def rotationMatrixYW(angle):
    return np.matrix([[1, 0, 0, 0], [0, np.cos(angle), 0, -np.sin(angle)], [0, 0, 1, 0], [np.sin(angle), 0, 0, np.cos(angle)]])

def rotateZW(x, y, z, w, angle):
    return x, y, z * np.cos(angle) - w * np.sin(angle), z * np.sin(angle) + w * np.cos(angle)

def rotationMatrixZW(angle):
    return np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(angle), -np.sin(angle)], [0, 0, np.sin(angle), np.cos(angle)]])


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

class fourSpace:
    def __init__(self, xN, yN, zN, wN, sideLength):
        self.xN = xN * sideLength
        self.yN = yN * sideLength
        self.zN = zN * sideLength
        self.wN = wN * sideLength
        self.gridLines = []
        self.gridLinesIndex = []

    def generateOutline(self):
        faces = [[[-self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, -self.wN/2]],
                [[-self.xN/2, self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, -self.wN/2]],
                [[self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, -self.wN/2]],
                [[self.xN/2, self.yN/2, -self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, -self.wN/2]],
                [[self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, -self.wN/2]],
                [[self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, self.wN/2]],
                [[self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, -self.wN/2]],
                [[self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, -self.wN/2]],
                [[-self.xN/2, self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, self.wN/2]],
                [[-self.xN/2, self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, -self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, self.yN/2, -self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, -self.zN/2, self.wN/2]],
                [[-self.xN/2, -self.yN/2, self.zN/2, -self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, self.yN/2, self.zN/2, -self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, -self.wN/2]],
                [[-self.xN/2, -self.yN/2, self.zN/2, self.wN/2], [-self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, self.yN/2, self.zN/2, self.wN/2], [self.xN/2, -self.yN/2, self.zN/2, self.wN/2]]]
        for i, face in enumerate(faces):
            self.gridLines.append(face)

    def drawGridLines(self):
        for line in self.gridLines:
            self.gridLinesIndex.append(canvas.create_line(
                xCoordinateToScreen(line[0][0]), yCoordinateToScreen(line[0][1]), xCoordinateToScreen(line[1][0]), yCoordinateToScreen(line[1][1]),
                xCoordinateToScreen(line[1][0]), yCoordinateToScreen(line[1][1]), xCoordinateToScreen(line[2][0]), yCoordinateToScreen(line[2][1]),
                xCoordinateToScreen(line[2][0]), yCoordinateToScreen(line[2][1]), xCoordinateToScreen(line[3][0]), yCoordinateToScreen(line[3][1]),
                xCoordinateToScreen(line[3][0]), yCoordinateToScreen(line[3][1]), xCoordinateToScreen(line[0][0]), yCoordinateToScreen(line[0][1]),
                fill='white', width=2, joinstyle='miter'))


resize = 2
root, canvas = createWindow(int(600 * resize), int(400 * resize))
main = overlay()

tesseracts = fourSpace(5, 6, 3, 2, 10)
tesseracts.generateOutline()
tesseracts.drawGridLines()
print(tesseracts.gridLinesIndex)



mainloop()
