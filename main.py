from tkinter import *
import numpy as np
import itertools


def createWindow(x: int, y: int):
    r = Tk()
    r.geometry(f"{x}x{y}")
    r.configure(background='black')
    r.title("Four Space")
    c = Canvas(r, width=x, height=y, bg='black')
    c.focus_set()
    c.pack()
    r.focus_set()
    return r, c

def tesseractPoints(x, y, z, w):
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

homogenize = lambda x: x[:2]/x[-1]

def translationMatrix(x, y, z, w):
    return np.array([[1, 0, 0, 0, x], [0, 1, 0, 0, y], [0, 0, 1, 0, z], [0, 0, 0, 1, w], [0, 0, 0, 0, 1]])

def rotationMatrixXY(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0, 0, 0], [np.sin(angle), np.cos(angle), 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

def rotationMatrixXZ(angle):
    return np.array([[np.cos(angle), 0, -np.sin(angle), 0, 0], [0, 1, 0, 0, 0], [np.sin(angle), 0, np.cos(angle), 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

def rotationMatrixXW(angle):
    return np.array([[np.cos(angle), 0, 0, -np.sin(angle), 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [np.sin(angle), 0, 0, np.cos(angle), 0], [0, 0, 0, 0, 1]])

def rotationMatrixYZ(angle):
    return np.array([[1, 0, 0, 0, 0], [0, np.cos(angle), -np.sin(angle), 0, 0], [0, np.sin(angle), np.cos(angle), 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

def rotationMatrixYW(angle):
    return np.array([[1, 0, 0, 0, 0], [0, np.cos(angle), 0, -np.sin(angle), 0], [0, 0, 1, 0, 0], [0, np.sin(angle), 0, np.cos(angle), 0], [0, 0, 0, 0, 1]])

def rotationMatrixZW(angle):
    return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, np.cos(angle), -np.sin(angle), 0], [0, 0, np.sin(angle), np.cos(angle), 0], [0, 0, 0, 0, 1]])


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
    def __init__(self, xN, yN, zN, wN, sideLength, screenX=180, screenY=126):
        self.xN = xN * sideLength
        self.yN = yN * sideLength
        self.zN = zN * sideLength
        self.wN = wN * sideLength
        self.sideLength = sideLength
        self.gridLines = []
        self.gridLinesIndex = []
        self.physicalObjects = set()
        self.currentTransform = np.identity(5)
        self.screenMatrix = np.array([[resize, 0, 0, 0, resize * screenX],
                                 [0, -resize, 0, 0, resize * screenY],
                                 [0, 0, resize, 0, 0],
                                 [0, 0, 0, resize, 0],
                                 [0, 0, 0, 0, 1]])

    def generateOutline(self):
        faces = tesseractPoints(self.xN/2, self.yN/2, self.zN/2, self.wN/2)
        for face in faces:
            self.gridLines.append(face)

    def drawGridLines(self):
        for face in self.gridLines:
            self.gridLinesIndex.append(canvas.create_polygon(
                *itertools.chain.from_iterable((self.screenMatrix @ self.currentTransform @ x)[:2] for x in face),
                outline='white', width=1, fill='', joinstyle=ROUND))


    def transform(self, matrix):
        self.currentTransform = matrix @ self.currentTransform
        for index, face in zip(self.gridLinesIndex, self.gridLines):
            canvas.coords(index,
            *itertools.chain.from_iterable((self.screenMatrix @ self.currentTransform @ x)[:2] for x in face))
        for hoi in self.physicalObjects:
            for index, _ in zip(hoi.indexes, hoi.geometry):
                canvas.coords(index,
            *itertools.chain.from_iterable((self.screenMatrix @ self.currentTransform @ x)[:2] for x in _))


class physicalObject:
    def __init__(self):
        self.geometry = []
        self.indexes = []
        self.space = None

    def addGeometry(self, geometry, indexes):
        self.geometry.append(geometry)
        self.indexes.append(indexes)
        self.update()
        
    def spawn(self, space):
        self.space = space
        space.physicalObjects.add(self)
        self.update()

    def update(self):
        if self.space:
            for index, _ in zip(self.indexes, self.geometry):
                canvas.coords(index,
            *itertools.chain.from_iterable((self.space.screenMatrix @ self.space.currentTransform @ x)[:2] for x in _))



resize = 2
root, canvas = createWindow(int(600 * resize), int(400 * resize))
main = overlay()

tesseracts = fourSpace(7, 7, 7, 7, 12)
tesseracts.generateOutline()
tesseracts.drawGridLines()
tesseracts.transform(rotationMatrixXW(-np.pi / 3) @ rotationMatrixXW(np.pi / 17))


def rotateBy12(event):
    tesseracts.transform(rotationMatrixXZ(np.pi / 64) @ rotationMatrixXY(np.pi / 32))

def rotateByNeg12(event):
    tesseracts.transform(rotationMatrixXW(-np.pi / 71) @ rotationMatrixYW(-np.pi / 64))

g = root.bind('w', rotateBy12)
b = root.bind('s', rotateByNeg12)
mainloop()


