from tkinter import *

width = 1024
height = 768


def startGame():
    """Opens the game window."""
    root = Tk()
    root.geometry("{}x{}".format(width, height))
    root.configure(background='black')
    root.title("Four Space")
    import screenModes
    screenModes.startScreen(root, width, height)

    mainloop()


if __name__ == '__main__':
    startGame()
