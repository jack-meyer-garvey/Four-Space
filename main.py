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


def xCoordinateToScreen(x: int):
    return resize * (180 + x)


def yCoordinateToScreen(y: int):
    return resize * (126 - y)


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

mainloop()
