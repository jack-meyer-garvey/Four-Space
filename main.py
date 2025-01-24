from tkinter import *


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
        self.textboxIndex = canvas.create_rectangle(34 * resize, 272 * resize, 566 * resize, 388 * resize,
                                                    outline='white', width=4 * resize, state=HIDDEN)
        self.textboxText = canvas.create_text(46 * resize, 278 * resize, text='', fill="white", anchor=NW,
                                              font=('System', int(12 * resize)), width=508 * resize)

resize = 2
root, canvas = createWindow(int(600 * resize), int(400 * resize))

main = overlay()
main.drawGridLeft(6, 6)
x, y = main.gridToScreen(1, 0)
b = canvas.create_rectangle(x, y, x + main.sideX, y + main.sideY, fill='medium sea green', width=2 * resize,
                            outline='white')
mainloop()
