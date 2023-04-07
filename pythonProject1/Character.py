import tkinterDrawings
from TextBox import *
from PIL import Image, ImageTk, ImageOps


def gainControl(stuck=False):
    """gives control of whichever character is meant to be controlled"""
    if Character.controlled is not None:
        Character.controlled.canvas.bind("<KeyPress-Up>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-Down>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-Left>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-Right>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-w>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-s>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-a>", Character.controlled.Press)
        Character.controlled.canvas.bind("<KeyPress-d>", Character.controlled.Press)
        if stuck is False:
            Character.controlled.canvas.bind("<KeyPress-1>", switchPerspective)
            Character.controlled.canvas.bind("<KeyPress-2>", switchPerspective)
            Character.controlled.canvas.bind("<KeyPress-3>", switchSlice)
            Character.controlled.canvas.bind("<KeyPress-4>", switchSlice)
        Character.controlled.canvas.bind("<KeyPress-BackSpace>", lambda event: rewind())
        Character.controlled.canvas.bind("<KeyPress-c>", lambda event: openMenu())
        Character.controlled.canvas.bind("<KeyPress-z>", lambda event: inspect())


def changeMenuText(text):
    if len(menuText) != 0:
        menuText.pop()
    menuText.append(text)


menuThings = []
menuImageReference = []
menuText = []


def openMenu():
    """Opens the menu"""

    img = PhotoImage(file="Pics/Menu.png")
    menuImageReference.append(img)
    box = textBox.canvas.create_image((512, 268), image=img)
    menuThings.append(box)
    if len(menuText) != 0:
        sentences = re.split('(?<=[.?!]\")(?=[\s$\"])|(?<=[.?!])(?=[\s$])', menuText[0])
        maxChar = 40
        for _ in range(len(sentences)):
            sentences[_] = textwrap.fill(sentences[_], maxChar)
        newText = ''
        for _ in sentences:
            if len(_) != 0:
                newText += _.strip() + '\n'
        this = textBox.canvas.create_text(348, 104, text=newText, fill="white", font="system 18",
                                          anchor=NW)
        menuThings.append(this)
    Character.canvas.bind("<KeyPress-c>", lambda event: closeMenu())


def closeMenu():
    """Closes the menu"""
    for _ in menuThings:
        Character.canvas.delete(_)
    menuThings.clear()
    Character.canvas.bind("<KeyPress-c>", lambda event: openMenu())


def loseControl(canvas, stuck=False):
    """unbinds all buttons used in character movement"""
    canvas.unbind("<KeyPress-Up>")
    canvas.unbind("<KeyPress-Down>")
    canvas.unbind("<KeyPress-Left>")
    canvas.unbind("<KeyPress-Right>")
    canvas.unbind("<KeyPress-w>")
    canvas.unbind("<KeyPress-s>")
    canvas.unbind("<KeyPress-a>")
    canvas.unbind("<KeyPress-d>")
    if stuck is False:
        canvas.unbind("<KeyPress-1>")
        canvas.unbind("<KeyPress-2>")
        canvas.unbind("<KeyPress-3>")
        canvas.unbind("<KeyPress-4>")
    canvas.unbind("<KeyPress-BackSpace>")


def rewind():
    closeMenu()
    for _ in Character.instances:
        if len(_.positionPath) > 1:
            _.positionPath.pop(-1)
            _.position = _.positionPath[-1][:]
            _.lookingPath.pop(-1)
            _.looking = _.lookingPath[-1]
    updateScreen()


def updateScreen():
    for _ in Character.instances:
        if _.position is not None:
            _.teleportChar()
    updateLasers()
    updateMirrors()


def updateLasers():
    """Deletes all drawn lasers and then draws all lasers after checking for perspective"""
    for _ in laser.lasersDrawn:
        Character.canvas.delete(_)
    laser.lasersDrawn.clear()

    positions = []
    for _ in Character.instances:
        if _.position is not None:
            positions.append(_.position)

    for L in laser.instances:
        L.drawLaser(positions)


def updateMirrors():
    for _ in mirror.instances:
        pass


def setPerspectives(Left, Right):
    """Changes the text associated with the left and right perspectives"""
    if Character.perspectiveTextIDRight is None:
        Character.perspectiveTextIDLeft = Character.canvas.create_text(256, 522,
                                                                       text="{} - {}".format(
                                                                           Character.perspectives[Left],
                                                                           Character.SliceText[Character.SliceLeft]),
                                                                       fill="white",
                                                                       font="system 18")
        Character.perspectiveLeft = Left
        Character.perspectiveTextIDRight = Character.canvas.create_text(768, 522,
                                                                        text="{} - {}".format(
                                                                            Character.perspectives[Right],
                                                                            Character.SliceText[Character.SliceRight]),
                                                                        fill="white",
                                                                        font="system 18")
        Character.perspectiveRight = Right
    else:
        Character.canvas.itemconfig(Character.perspectiveTextIDLeft,
                                    text="{} - {}".format(
                                        Character.perspectives[Left],
                                        Character.SliceText[Character.SliceLeft]))
        Character.canvas.itemconfig(Character.perspectiveTextIDRight,
                                    text="{} - {}".format(
                                        Character.perspectives[Right],
                                        Character.SliceText[Character.SliceRight]))


def switchPerspective(event):
    """Switches one perspective shown on screen and then teleports the characters to the right spot"""
    if event.keysym == '1':
        Character.perspectiveLeft += 1
        if Character.perspectiveLeft > 6:
            Character.perspectiveLeft = 1
    if event.keysym == '2':
        Character.perspectiveRight += 1
        if Character.perspectiveRight > 6:
            Character.perspectiveRight = 1
    setPerspectives(Character.perspectiveLeft, Character.perspectiveRight)
    updateScreen()


def manualPerspectiveChange(Left, Right):
    """Sets the perspective through a function call rather than an event."""
    setPerspectives(Left, Right)
    Character.perspectiveLeft = Left
    Character.perspectiveRight = Right
    updateScreen()


def switchSlice(event):
    """Toggles the perspective type."""
    if event is None:
        Character.SliceLeft = (Character.SliceLeft + 1) % 2
        Character.SliceRight = (Character.SliceRight + 1) % 2
    else:
        if event.keysym == '3':
            Character.SliceLeft = (Character.SliceLeft + 1) % 2
        if event.keysym == '4':
            Character.SliceRight = (Character.SliceRight + 1) % 2
    setPerspectives(Character.perspectiveLeft, Character.perspectiveRight)
    updateScreen()


def setSlice(Left, Right):
    Character.SliceLeft = Left
    Character.SliceRight = Right
    setPerspectives(Character.perspectiveLeft, Character.perspectiveRight)
    updateScreen()


def distance(position1, position2):
    """Calculates the distance between a position and a character"""
    dist = 0
    for _ in range(len(position1)):
        dist += (position1[_] - position2[_]) ** 2
    return dist ** 0.5


def inspect():
    if Character.controlled is not None:
        for _ in Character.instances:
            if _.position is not None:
                if tuple(_.position) == tuple(
                        sum(x) for x in zip(tuple(Character.controlled.position), Character.controlled.looking)):
                    for __ in _.inspectFunctions:
                        __()
                    loseControl(Character.canvas, stuck=True)
                    textBox.exitFunctions.append(lambda: gainControl(stuck=True))
                    runQueue()


class Character:
    instances = []
    controlled = None
    # Keeps track of grid perspectives
    perspectiveLeft = None
    SliceLeft = 0
    perspectiveRight = None
    SliceRight = 0
    SliceText = {0: "Projection", 1: "Slice"}
    perspectives = {1: "XY", 2: "ZW", 3: "XZ", 4: "XW", 5: "YZ", 6: "YW"}
    perspectiveKey = {1: [0, 1], 2: [2, 3], 3: [0, 2], 4: [0, 3], 5: [1, 2], 6: [1, 3]}
    perspectiveKeyInverse = {1: [2, 3], 2: [0, 1], 3: [1, 3], 4: [1, 2], 5: [0, 3], 6: [0, 2]}
    perspectiveTextIDLeft = None
    perspectiveTextIDRight = None
    # Prerogatives
    winCon = []
    winFunc = []
    flags = []
    # root and canvas the characters are on
    root = None
    canvas = None
    # Keeps track of loops
    loopsRunning = []

    def __init__(self, inspectFunctions=[]):
        Character.instances.append(self)
        # tkinter stuff
        # position in 4 space
        self.position = None
        # Keeps track of all positions the object has been in since spawning
        self.positionPath = []
        self.lookingPath = []
        # dimension of the n by n grid that object is in
        self.n = None
        # Picture used for the grid
        self.picture = None
        self.imageID_Right = None
        self.imageID_Left = None
        # Boolean to see if an object can be pushed
        self.push = True
        self.stop = False
        self.slide = False
        # Defines the direction a character is facing
        self.looking = (0, 0, 0, 0)
        # Pictures for animations
        self.frames = []
        self.frameNum = 0
        # Text boxes or other functions that play when inspected
        self.inspectFunctions = []
        for _ in inspectFunctions:
            self.inspectFunctions.append(_)

    def spawnChar(self, xpos, ypos, zpos, wpos, n):
        """Draws a character in the 4D discrete tesseract grid with side length n"""
        self.n = n
        self.position = [xpos, ypos, zpos, wpos]
        self.positionPath.clear()
        self.positionPath.append(self.position[:])
        self.lookingPath.clear()
        self.lookingPath.append(self.looking)

        # See https://github.com/python-pillow/Pillow/issues/6020
        ImageTk.PhotoImage("RGB").paste(Image.new("RGB", (1, 1)))

        pic = ImageTk.getimage(self.picture)
        self.picture = ImageTk.PhotoImage(pic.resize((int(pic.width * 8 / self.n), int(pic.height * 8 / self.n))))

        frames = []
        for _ in self.frames:
            pic = ImageTk.getimage(_)
            frames.append(ImageTk.PhotoImage(pic.resize((int(pic.width * 8 / self.n), int(pic.height * 8 / self.n)))))
        self.frames = frames

        # Checks which perspective has been set, and then draws accordingly
        thing = tkinterDrawings.grid_to_screen(self.position[Character.perspectiveKey[Character.perspectiveLeft][0]],
                                               self.position[Character.perspectiveKey[Character.perspectiveLeft][1]],
                                               n, "Left")
        self.imageID_Right = Character.canvas.create_image(thing[0], thing[1], image=self.picture)

        thing = tkinterDrawings.grid_to_screen(self.position[Character.perspectiveKey[Character.perspectiveRight][0]],
                                               self.position[Character.perspectiveKey[Character.perspectiveRight][1]],
                                               n, "Right")
        self.imageID_Left = Character.canvas.create_image(thing[0], thing[1], image=self.picture)
        updateScreen()

    def teleportChar(self):
        """Places the Character in the grid at its current position while accounting for perspective"""
        thing = tkinterDrawings.grid_to_screen(self.position[Character.perspectiveKey[Character.perspectiveLeft][0]],
                                               self.position[Character.perspectiveKey[Character.perspectiveLeft][1]],
                                               self.n, "Left")
        Character.canvas.coords(self.imageID_Left, thing[0], thing[1])

        if Character.SliceLeft == 1:
            Character.canvas.itemconfig(self.imageID_Left, state='hidden')
            if self.position[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]] \
                    == Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]] \
                    and self.position[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]] \
                    == Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]:
                Character.canvas.itemconfig(self.imageID_Left, state='normal')
        else:
            Character.canvas.itemconfig(self.imageID_Left, state='normal')

        thing = tkinterDrawings.grid_to_screen(self.position[Character.perspectiveKey[Character.perspectiveRight][0]],
                                               self.position[Character.perspectiveKey[Character.perspectiveRight][1]],
                                               self.n, "Right")
        Character.canvas.coords(self.imageID_Right, thing[0], thing[1])
        if Character.SliceRight == 1:
            Character.canvas.itemconfig(self.imageID_Right, state='hidden')
            if self.position[Character.perspectiveKeyInverse[Character.perspectiveRight][0]] \
                    == Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveRight][0]] \
                    and self.position[Character.perspectiveKeyInverse[Character.perspectiveRight][1]] \
                    == Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveRight][1]]:
                Character.canvas.itemconfig(self.imageID_Right, state='normal')
        else:
            Character.canvas.itemconfig(self.imageID_Right, state='normal')

    def deSpawnChar(self):
        Character.canvas.delete(self.imageID_Right)
        Character.canvas.delete(self.imageID_Left)
        self.positionPath.clear()
        self.lookingPath.clear()
        self.looking = (0, 0, 0, 0)
        self.position = None

    def animate(self, frameRate):
        self.frameNum += 1
        if self.frameNum == len(self.frames):
            self.frameNum = 0
        Character.canvas.itemconfig(self.imageID_Right, image=self.frames[self.frameNum])
        Character.canvas.itemconfig(self.imageID_Left, image=self.frames[self.frameNum])
        loop = self.root.after(frameRate, self.animate, frameRate)
        Character.loopsRunning.append(loop)

    def flip(self):
        self.picture = ImageOps.flip(self.picture)
        for _ in range(len(self.frames)):
            self.frames[_] = ImageOps.flip(self.frames[_])

    def move(self, x, y, z, w, press=False, stop=False):
        """Changes the character's position and visual in the perspective, then teleports"""
        self.position[Character.perspectiveKey[Character.perspectiveLeft][0]] += x
        self.position[Character.perspectiveKey[Character.perspectiveLeft][1]] += y
        self.position[Character.perspectiveKey[Character.perspectiveRight][0]] += z
        self.position[Character.perspectiveKey[Character.perspectiveRight][1]] += w
        stop = stop
        if press is True:
            borp = (
                Character.controlled.position[0] - Character.controlled.positionPath[-1][0],
                Character.controlled.position[1] - Character.controlled.positionPath[-1][1],
                Character.controlled.position[2] - Character.controlled.positionPath[-1][2],
                Character.controlled.position[3] - Character.controlled.positionPath[-1][3])
            if borp != (0, 0, 0, 0):
                Character.controlled.looking = borp
        # Checks to make sure you are still in the grid
        for _ in range(len(self.position)):
            if self.position[_] <= 0:
                self.move(-x, -y, -z, -w, stop=True)
                stop = True
            if self.position[_] > self.n:
                self.move(-x, -y, -z, -w, stop=True)
                stop = True
        # Checks to see if you are colliding with a pushable object
        for _ in Character.instances:
            if _ != self and _.position == self.position:
                if _.push and not _.stop:
                    if stop is True:
                        _.move(x, y, z, w, stop=True)
                    else:
                        _.move(x, y, z, w)
                    stop = True
                if _.stop:
                    self.move(-x, -y, -z, -w, stop=True)
                    stop = True
        # Checks if you are sliding and haven't been stopped
        if stop is False and self.slide is True:
            self.move(x, y, z, w)

    def Press(self, event):
        """Each direction moves the character once in whatever perspective is currently visible"""
        # Change
        closeMenu()
        if event.keysym == 'Up':
            self.move(0, 0, 0, 1, press=True)
        if event.keysym == 'Down':
            self.move(0, 0, 0, -1, press=True)
        if event.keysym == 'Left':
            self.move(0, 0, -1, 0, press=True)
        if event.keysym == 'Right':
            self.move(0, 0, 1, 0, press=True)
        if event.keysym == 'w':
            self.move(0, 1, 0, 0, press=True)
        if event.keysym == 's':
            self.move(0, -1, 0, 0, press=True)
        if event.keysym == 'a':
            self.move(-1, 0, 0, 0, press=True)
        if event.keysym == 'd':
            self.move(1, 0, 0, 0, press=True)

        # saves the position path, but first it checks to make sure something about the position has actually changed after the last move
        stuff = []
        things = []
        lookingstuff = []
        lookingthings = []
        for _ in Character.instances:
            if _.position is not None:
                stuff.append(_.positionPath[-1])
                things.append(_.position)
                lookingstuff.append(_.lookingPath[-1])
                lookingthings.append(_.looking)
        if stuff != things or lookingstuff != lookingthings:
            for _ in Character.instances:
                if _.position is not None:
                    _.positionPath.append(_.position[:])
                    borp = (
                        _.positionPath[-1][0] - _.positionPath[-2][0], _.positionPath[-1][1] - _.positionPath[-2][1],
                        _.positionPath[-1][2] - _.positionPath[-2][2], _.positionPath[-1][3] - _.positionPath[-2][3])
                    if borp != (0, 0, 0, 0):
                        _.looking = borp
                    _.lookingPath.append(_.looking)

        if all(win() for win in Character.winCon):
            Func = Character.winFunc.copy()
            Character.winCon.clear()
            Character.winFunc.clear()
            for _ in Func:
                _()

        for _ in Character.flags:
            if _[0]():
                Character.flags.remove(_)
                for __ in _:
                    __()

        updateScreen()

    def manualPress(self, x, y, z, w):
        self.move(x, y, z, w)
        # saves the position path, but first it checks to make sure something about the position has actually changed after the last move
        stuff = []
        things = []
        lookingstuff = []
        lookingthings = []
        for _ in Character.instances:
            if _.position is not None:
                stuff.append(_.positionPath[-1])
                things.append(_.position)
                lookingstuff.append(_.lookingPath[-1])
                lookingthings.append(_.looking)
        if stuff != things or lookingstuff != lookingthings:
            for _ in Character.instances:
                if _.position is not None:
                    _.positionPath.append(_.position[:])
                    borp = (
                        _.positionPath[-1][0] - _.positionPath[-2][0], _.positionPath[-1][1] - _.positionPath[-2][1],
                        _.positionPath[-1][2] - _.positionPath[-2][2], _.positionPath[-1][3] - _.positionPath[-2][3])
                    if borp != (0, 0, 0, 0):
                        _.looking = borp
                    _.lookingPath.append(_.looking)

        if all(win() for win in Character.winCon):
            Func = Character.winFunc.copy()
            Character.winCon.clear()
            Character.winFunc.clear()
            for _ in Func:
                _()

        for _ in range(len(Character.flags)):
            if Character.flags[_][0]():
                for __ in Character.flags.pop(_):
                    __()

        updateScreen()


def between(x, num1, num2):
    return (num1 <= x <= num2) or (num1 >= x >= num2)


class laser(Character):
    instances = []
    lasersDrawn = []

    def __init__(self, direction):
        super().__init__()
        laser.instances.append(self)
        self.direction = direction
        self.picture = PhotoImage(file="Pics/Laser.png")

    def drawLaser(self, positions, position=None, direction=None, color='red'):
        """Draws a laser that starts at (x, y)"""
        for side in ["Left", "Right"]:
            if position is None:
                position = self.position
            if direction is None:
                direction = self.direction
            point = [None, None]
            notPoint = [None, None]
            closestBlock = None
            for _ in positions:
                if _ != position:
                    equations = []
                    zeros = []
                    for __ in range(len(position)):
                        if direction[__] != 0 and ((_[__] - position[__] > 0) ==
                                                   (direction[__] > 0) or (_[__] == position[__] > 0)):
                            equations.append((_[__] - position[__]) / direction[__])
                        else:
                            zeros.append(__)
                    if len(equations) != 0:
                        if equations.count(equations[0]) == len(equations) and all(
                                _[__] == position[__] for __ in zeros):
                            if closestBlock is None:
                                closestBlock = _
                            if distance(closestBlock, position) >= distance(_, position):
                                closestBlock = _
                                if side == 'Left':
                                    point = [_[Character.perspectiveKey[Character.perspectiveLeft][0]],
                                             _[Character.perspectiveKey[Character.perspectiveLeft][1]]]
                                    notPoint = [_[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]],
                                                _[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]]
                                if side == 'Right':
                                    point = [_[Character.perspectiveKey[Character.perspectiveRight][0]],
                                             _[Character.perspectiveKey[Character.perspectiveRight][1]]]
                                    notPoint = [_[Character.perspectiveKeyInverse[Character.perspectiveRight][0]],
                                                _[Character.perspectiveKeyInverse[Character.perspectiveRight][1]]]
            if side == 'Left':
                x = position[Character.perspectiveKey[Character.perspectiveLeft][0]]
                y = position[Character.perspectiveKey[Character.perspectiveLeft][1]]
                dx = direction[Character.perspectiveKey[Character.perspectiveLeft][0]]
                dy = direction[Character.perspectiveKey[Character.perspectiveLeft][1]]
                notx = position[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]]
                noty = position[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]
                notdx = direction[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]]
                notdy = direction[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]
            if side == 'Right':
                x = position[Character.perspectiveKey[Character.perspectiveRight][0]]
                y = position[Character.perspectiveKey[Character.perspectiveRight][1]]
                dx = direction[Character.perspectiveKey[Character.perspectiveRight][0]]
                dy = direction[Character.perspectiveKey[Character.perspectiveRight][1]]
                notx = position[Character.perspectiveKeyInverse[Character.perspectiveRight][0]]
                noty = position[Character.perspectiveKeyInverse[Character.perspectiveRight][1]]
                notdx = direction[Character.perspectiveKeyInverse[Character.perspectiveRight][0]]
                notdy = direction[Character.perspectiveKeyInverse[Character.perspectiveRight][1]]
            if point == [None, None]:

                if dx > 0:
                    point[0] = self.n + 0.5
                    point[1] = dy / dx * (self.n + 0.5 - x) + y
                if dx < 0:
                    point[0] = 0.5
                    point[1] = dy / dx * (0.5 - x) + y
                if point[1] is None or point[1] > self.n + 0.5 or point[1] < 0:
                    point = [None, None]
                    if dy > 0:
                        point[1] = self.n + 0.5
                        point[0] = dx / dy * (self.n + 0.5 - y) + x
                    if dy < 0:
                        point[1] = 0.5
                        point[0] = dx / dy * (0.5 - y) + x
                if point[0] is None and point[1] is None:
                    point = [x, y]
                ###
                if notdx > 0:
                    notPoint[0] = self.n + 0.5
                    notPoint[1] = notdy / notdx * (self.n + 0.5 - notx) + noty
                if notdx < 0:
                    notPoint[0] = 0.5
                    notPoint[1] = notdy / notdx * (0.5 - notx) + noty
                if notPoint[1] is None or notPoint[1] > self.n + 0.5 or notPoint[1] < 0:
                    notPoint = [None, None]
                    if notdy > 0:
                        notPoint[1] = self.n + 0.5
                        notPoint[0] = notdx / notdy * (self.n + 0.5 - noty) + notx
                    if notdy < 0:
                        notPoint[1] = 0.5
                        notPoint[0] = notdx / notdy * (0.5 - noty) + notx
                if notPoint[0] is None and notPoint[1] is None:
                    notPoint = [notx, noty]
            ###
            thing1 = tkinterDrawings.grid_to_screen(x, y, self.n, side)
            thing2 = tkinterDrawings.grid_to_screen(point[0], point[1], self.n, side)

            line = None
            if side == "Left":
                if Character.SliceLeft == 0:
                    line = Character.canvas.create_line(thing1[0], thing1[1], thing2[0], thing2[1], fill=color, width=4)
                if Character.SliceLeft == 1:
                    if [Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]],
                        Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]] \
                            == notPoint:
                        if [Character.controlled.position[
                                Character.perspectiveKeyInverse[Character.perspectiveLeft][0]],
                            Character.controlled.position[
                                Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]] \
                                == [notx, noty]:
                            line = Character.canvas.create_line(thing1[0], thing1[1], thing2[0], thing2[1], fill=color,
                                                                width=4)
                        else:
                            line = Character.canvas.create_rectangle(thing2[0] - 2, thing2[1] - 2, thing2[0] + 2,
                                                                     thing2[1] + 2,
                                                                     fill=color, outline=color)
                    elif between(
                            Character.controlled.position[
                                Character.perspectiveKeyInverse[Character.perspectiveLeft][0]],
                            notPoint[0],
                            position[Character.perspectiveKeyInverse[Character.perspectiveLeft][0]]) and \
                            between(Character.controlled.position[
                                        Character.perspectiveKeyInverse[Character.perspectiveLeft][1]],
                                    notPoint[1],
                                    position[Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]):
                        t1 = None
                        t2 = None
                        if direction[
                            Character.perspectiveKeyInverse[Character.perspectiveLeft][0]] != 0:
                            t1 = (Character.controlled.position[
                                      Character.perspectiveKeyInverse[Character.perspectiveLeft][0]] -
                                  position[
                                      Character.perspectiveKeyInverse[Character.perspectiveLeft][0]]) / direction[
                                     Character.perspectiveKeyInverse[Character.perspectiveLeft][0]]
                        if direction[
                            Character.perspectiveKeyInverse[Character.perspectiveLeft][1]] != 0:
                            t2 = (Character.controlled.position[
                                      Character.perspectiveKeyInverse[Character.perspectiveLeft][1]] -
                                  position[
                                      Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]) / direction[
                                     Character.perspectiveKeyInverse[Character.perspectiveLeft][1]]
                        if t1 is None or t1 == t2:
                            point = tkinterDrawings.grid_to_screen(position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][0]] + t2 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][0]],
                                                                   position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][1]] + t2 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][1]], self.n,
                                                                   "Left"
                                                                   )
                            line = Character.canvas.create_rectangle(point[0] - 2, point[1] - 2, point[0] + 2,
                                                                     point[1] + 2,
                                                                     fill=color, outline=color)

                        if t2 is None:
                            point = tkinterDrawings.grid_to_screen(position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][0]] + t1 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][0]],
                                                                   position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][1]] + t1 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveLeft][1]], self.n,
                                                                   "Left"
                                                                   )
                            line = Character.canvas.create_rectangle(point[0] - 2, point[1] - 2, point[0] + 2,
                                                                     point[1] + 2,
                                                                     fill=color, outline=color)

            laser.lasersDrawn.append(line)
            if side == "Right":
                if Character.SliceRight == 0:
                    line = Character.canvas.create_line(thing1[0], thing1[1], thing2[0], thing2[1], fill=color, width=4)
                if Character.SliceRight == 1:
                    if [Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveRight][0]],
                        Character.controlled.position[Character.perspectiveKeyInverse[Character.perspectiveRight][1]]] \
                            == notPoint:
                        if [Character.controlled.position[
                                Character.perspectiveKeyInverse[Character.perspectiveRight][0]],
                            Character.controlled.position[
                                Character.perspectiveKeyInverse[Character.perspectiveRight][1]]] \
                                == [notx, noty]:
                            line = Character.canvas.create_line(thing1[0], thing1[1], thing2[0], thing2[1], fill=color,
                                                                width=4)
                        else:
                            line = Character.canvas.create_rectangle(thing2[0] - 2, thing2[1] - 2, thing2[0] + 2,
                                                                     thing2[1] + 2,
                                                                     fill=color, outline=color)
                    elif between(
                            Character.controlled.position[
                                Character.perspectiveKeyInverse[Character.perspectiveRight][0]],
                            notPoint[0],
                            position[Character.perspectiveKeyInverse[Character.perspectiveRight][0]]) and \
                            between(Character.controlled.position[
                                        Character.perspectiveKeyInverse[Character.perspectiveRight][1]],
                                    notPoint[1],
                                    position[Character.perspectiveKeyInverse[Character.perspectiveRight][1]]):
                        t1 = None
                        t2 = None
                        if direction[
                            Character.perspectiveKeyInverse[Character.perspectiveRight][0]] != 0:
                            t1 = (Character.controlled.position[
                                      Character.perspectiveKeyInverse[Character.perspectiveRight][0]] -
                                  position[
                                      Character.perspectiveKeyInverse[Character.perspectiveRight][0]]) / direction[
                                     Character.perspectiveKeyInverse[Character.perspectiveRight][0]]
                        if direction[
                            Character.perspectiveKeyInverse[Character.perspectiveRight][1]] != 0:
                            t2 = (Character.controlled.position[
                                      Character.perspectiveKeyInverse[Character.perspectiveRight][1]] -
                                  position[
                                      Character.perspectiveKeyInverse[Character.perspectiveRight][1]]) / direction[
                                     Character.perspectiveKeyInverse[Character.perspectiveRight][1]]
                        if t1 is None or t1 == t2:
                            point = tkinterDrawings.grid_to_screen(position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][0]] + t2 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][0]],
                                                                   position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][1]] + t2 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][1]], self.n,
                                                                   "Right"
                                                                   )
                            line = Character.canvas.create_rectangle(point[0] - 2, point[1] - 2, point[0] + 2,
                                                                     point[1] + 2,
                                                                     fill=color, outline=color)

                        if t2 is None:
                            point = tkinterDrawings.grid_to_screen(position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][0]] + t1 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][0]],
                                                                   position[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][1]] + t1 *
                                                                   direction[
                                                                       Character.perspectiveKey[
                                                                           Character.perspectiveRight][1]], self.n,
                                                                   "Right"
                                                                   )
                            line = Character.canvas.create_rectangle(point[0] - 2, point[1] - 2, point[0] + 2,
                                                                     point[1] + 2,
                                                                     fill=color, outline=color)

            laser.lasersDrawn.append(line)


class mirror(Character):
    instances = []

    def __init__(self, direction):
        super().__init__()
        mirror.instances.append(self)
        self.direction = direction
        self.picture = PhotoImage(file="Pics/glassSphere.png")
