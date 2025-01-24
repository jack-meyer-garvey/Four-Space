from tkinter import *
from collections import deque
import textwrap


def createWindow(x: int, y: int):
    r = Tk()
    r.geometry(f"{x}x{y}")
    r.configure(background='black')
    r.title("Four Space")
    c = Canvas(r, width=x, height=y, bg='black')
    c.focus_set()
    c.pack()
    return r, c


def drawRectangularGrid(cornerX: int, cornerY: int, nX: int, nY: int, lengthX: int, lengthY: int, outline=4, grid=2,
                        scale: int = None):
    if scale is not None:
        cornerX, cornerY, lengthX, lengthY, outline, grid = \
            scale * cornerX, scale * cornerY, scale * lengthX, scale * lengthY, scale * outline, scale * grid
    iDs = list()
    sideX = lengthX / nX
    for x in range(nX - 1):
        iDs.append(canvas.create_line(cornerX + sideX * (x + 1), cornerY,
                                      cornerX + sideX * (x + 1), cornerY + lengthY,
                                      fill='white', width=grid))
    sideY = lengthY / nY
    for y in range(nY - 1):
        iDs.append(canvas.create_line(cornerX, cornerY + sideY * (y + 1),
                                      cornerX + lengthX, cornerY + sideY * (y + 1),
                                      fill='white', width=grid))
    iDs.append(canvas.create_rectangle(cornerX, cornerY, cornerX + lengthX, cornerY + lengthY,
                                       outline='white', width=outline))
    return iDs, sideX, sideY


class overlay:
    def __init__(self, nX: int = 8, nY: int = 8):
        # memory for grid system
        self.sideX = 0
        self.sideY = 0
        # initialize indices
        self.leftGridIndex = None
        self.rightGridIndex = None
        self.textboxIndex = None
        self.textboxText = None
        self.drawGridLeft(nX, nY)
        self.drawRightBox()
        self.drawTextbox()
        # memory associated with resolving textbox Text
        self.textLines = deque()
        self.typingLoop = None
        self.lenLineShown = 0
        self.typingSpeed = 31

    def drawGridLeft(self, nX, nY):
        if self.leftGridIndex is not None:
            for _ in self.leftGridIndex:
                canvas.delete(_)
        self.leftGridIndex, self.sideX, self.sideY = drawRectangularGrid(34, 12, nX, nY, 248, 248, scale=resize)

    def gridToScreen(self, x, y):
        return 34 * resize + x * self.sideX, 12 * resize + y * self.sideY

    def drawRightBox(self):
        self.rightGridIndex = canvas.create_rectangle(318 * resize, 12 * resize, 566 * resize, 260 * resize,
                                                      outline='white', width=4 * resize)

    def drawTextbox(self):
        # noinspection PyTypeChecker
        self.textboxIndex = canvas.create_rectangle(34 * resize, 272 * resize, 566 * resize, 388 * resize,
                                                    outline='white', width=4 * resize, state=HIDDEN)
        self.textboxText = canvas.create_text(46 * resize, 278 * resize, text='', fill="white", anchor=NW,
                                              font=('System', int(12 * resize)), width=508 * resize)

    def runTextboxFile(self, fileName: str):
        canvas.itemconfigure(self.textboxIndex, state=NORMAL)
        for line in open(fileName).readlines():
            if line[0] == '#':
                self.textLines.append(textwrap.fill(line[1:], 50))
            elif line[:8] == 'command:':
                print(line[8:])
            else:
                self.textLines.append(textwrap.fill(line, 50))
        self.nextLine()

    def nextLine(self):
        if not self.typingLoop:
            if self.textLines:
                self.typeLetter()
            else:
                canvas.itemconfigure(self.textboxIndex, state=HIDDEN)
                canvas.itemconfigure(self.textboxText, text="")
                self.lenLineShown = 0

    def typeLetter(self):
        if self.lenLineShown != len(self.textLines[0]):
            self.typingLoop = root.after(self.typingSpeed, self.typeLetter)
            self.lenLineShown += 1
            canvas.itemconfigure(self.textboxText, text=self.textLines[0][:self.lenLineShown])
        else:
            self.typingLoop = None
            self.lenLineShown = 0
            self.textLines.popleft()

    def skipTyping(self):
        if self.typingLoop:
            self.lenLineShown = len(self.textLines[0]) - 1


resize = 1
root, canvas = createWindow(int(600 * resize), int(400 * resize))

main = overlay()
main.drawGridLeft(6, 6)
x, y = main.gridToScreen(3, 5)
b = canvas.create_rectangle(x, y, x + main.sideX, y + main.sideY, fill='medium sea green', width=2 * resize,
                            outline='white')
canvas.bind('<Button-1>', lambda event: main.nextLine())
canvas.bind('<Button-3>', lambda event: main.skipTyping())
canvas.focus_set()

mainloop()
