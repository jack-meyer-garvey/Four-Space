from Character import *
import tkinterDrawings
import playsound

loopsRunning = []
savedState = {}


def clearWindow(root):
    """Destroys all widgets and ends all loops (including loops within text boxes)"""
    for loop in loopsRunning:
        root.after_cancel(loop)
    for loop in Character.loopsRunning:
        root.after_cancel(loop)
    loopsRunning.clear()
    Character.loopsRunning.clear()
    clearText()
    loopsRunning.clear()
    for widget in root.winfo_children():
        widget.destroy()
    Character.perspectiveTextIDRight = None
    Character.perspectiveTextIDLeft = None
    Character.instances.clear()
    textBox.imageInstances.clear()
    textBox.instances.clear()
    textBox.imagesShown.clear()
    textBox.textQueue.clear()
    textBox.questionQueue.clear()
    textBox.closeFunctions.clear()
    textBox.exitFunctions.clear()
    textBox.loopsRunning.clear()
    textBox.textShown.clear()
    textBox.arrowBindings.clear()
    Character.flags.clear()
    Character.winCon.clear()
    Character.winFunc.clear()


def addSavedState(key, info):
    savedState[key] = info


def textFlicker(root, canvas, text):
    """Switches the color of text from white to black every 0.5 seconds"""
    if canvas.itemcget(text, 'fill') == 'white':
        canvas.itemconfigure(text, fill='black')
    else:
        canvas.itemconfigure(text, fill='white')
    loop = root.after(500, textFlicker, root, canvas, text)
    loopsRunning.append(loop)


def startScreen(root, width, height):
    """Black Screen that leads to the intro."""
    clearWindow(root)
    canvas = Canvas(root, width=width, height=height, bg='black')
    text = canvas.create_text(width / 2, height / 2, text="Press Z to start", fill="white", font="system 18")
    canvas.pack()
    textFlicker(root, canvas, text)
    canvas.bind("<KeyPress-z>", lambda event: tutorial(root, width, height))
    canvas.focus_set()


def tutorial(root, width, height):
    """Black Screen with the full control scheme labeled"""
    clearWindow(root)
    canvas = Canvas(root, width=width, height=height, bg='black')
    Character.root = root
    Character.canvas = canvas
    You = Character()
    Crate = Character()
    Crate2 = Character()
    Ball = Character()

    Goal = Character()
    Goal.picture = PhotoImage(file="Pics/Goal.png")
    Goal.push = False

    You.picture = PhotoImage(file="Pics/Blue Square.png")
    Crate.picture = PhotoImage(file="Pics/Crate.png")
    Crate2.picture = PhotoImage(file="Pics/Crate.png")
    Ball.picture = PhotoImage(file="Pics/Ball.png")

    Character.controlled = You
    textBox.root = root
    textBox.canvas = canvas

    textBox("Hello!")
    textBox("Welcome to Four Space!")
    textBox("Would you like to skip the tutorial?", options=['No', 'Yes'])
    textBox("...", condition=2,
            closeFunctions=[lambda: titleScreen(root, width, height), lambda: addSavedState("tutorial", "Skipped")])
    textBox("Press Z to advance the text.", closeFunctions=[lambda: addSavedState("tutorial", "Played")])
    runQueue()
    textBox("You already knew how to advance text, but I said it anyways...")
    textBox("I threw you into the game with no knowledge of the controls, and you figured one of them "
            "out on "
            "your own. Nice!")
    textBox("That teaching technique is called \"show, don't tell\", and it is a great way to make "
            "a fun video game!...")
    textBox("I will not be doing much of that.")
    textBox("Instead, I will explain everything in excruciating detail.")
    textBox("Press C for a reminder of the controls. ... Just don't forget the C key.",
            closeFunctions=[lambda: canvas.bind("<KeyPress-c>", lambda event: openMenu()),
                            lambda: changeMenuText("Z: advance text")])
    textBox("Remember, C stands for \"Can I please remember the other controls?\"")
    textBox("Press X to skip through text. This is especially helpful for text that takes a really "
            "really long time to be written out.......", textSpeed=60,
            closeFunctions=[lambda: changeMenuText("Z, X: advance, skip text")])
    textBox("However, don't always press X because it can ruin my comedic timing.")
    textBox(".......................... .......................... .......................... "
            ".......................... ..........................",
            textSpeed=60)
    textBox("Anyways, lets get started. Use the arrow keys to move.", closeFunctions=[
        lambda: setPerspectives(1, 1),
        lambda: Character.controlled.canvas.bind("<KeyPress-Up>", Character.controlled.Press),
        lambda: Character.controlled.canvas.bind("<KeyPress-Down>", Character.controlled.Press),
        lambda: Character.controlled.canvas.bind("<KeyPress-Left>", Character.controlled.Press),
        lambda: Character.controlled.canvas.bind("<KeyPress-Right>", Character.controlled.Press),
        lambda: tkinterDrawings.drawGrids(canvas, 8),
        lambda: You.spawnChar(4, 4, 4, 4, 8),
        lambda: changeMenuText("Z, X: advance, skip text. Arrow keys: Move")
    ])
    textBox("From this perspective, you look like a cute little blue square.")
    textBox("In reality, you are a cute little blue tesseract.")
    textBox("Of course, even tesseracts can make mistakes.")
    textBox("Press backspace to undo an action.",
            closeFunctions=[lambda: Character.controlled.canvas.bind("<KeyPress-BackSpace>", lambda event: rewind()),
                            lambda: changeMenuText("Z, X: advance, skip text. Arrow keys: Move. Backspace: Rewind.")])
    textBox("This world may seem boring right now, but it won't take much more to overwhelm you.")
    textBox("Here is a crate.", closeFunctions=[lambda: Crate.spawnChar(3, 4, 4, 4, 8)])
    textBox("Here is another... ... crate.", closeFunctions=[lambda: Crate2.spawnChar(6, 4, 2, 7, 8),
                                                             lambda: canvas.tag_raise(You.imageID_Left),
                                                             lambda: canvas.tag_raise(You.imageID_Right)])
    textBox("Try pushing one of them into the top right corner.")
    textBox("...", closeFunctions=[lambda: closeBoxes(),
                                   lambda: clearText(),
                                   lambda: Goal.spawnChar(8, 8, 4, 4, 8),
                                   lambda: canvas.tag_raise(You.imageID_Left),
                                   lambda: canvas.tag_raise(You.imageID_Right),
                                   lambda: Character.winCon.append(
                                       lambda: Crate.position == Goal.position),
                                   lambda: Character.winFunc.append(runQueue),
                                   lambda: Character.winFunc.append(Goal.deSpawnChar),
                                   lambda: changeMenuText(
                                       "Z, X: advance, skip text. Arrow keys: Move. Backspace: Rewind. Goal: Push box"
                                       " to (8, 8)")])
    textBox("Good job! You just solved your first puzzle in four space!", closeFunctions=[
        lambda: changeMenuText(
            "Z, X: advance, skip text. Arrow keys: Move. Backspace: Rewind.")
    ])
    textBox("You may have noticed that the crate located at (6, 4) is impossible to push. ... Im-push-ible.")
    textBox("That is because you don't have all the information yet.")
    textBox("While it is true in some sense that the crate is located at (6, 4), it would be more accurate"
            " to say that it is located at (6, 4, 2, 7).")
    textBox("Here is another perspective. Notice that the left XY plane has now been changed to show the"
            " ZW plane.", closeFunctions=[lambda: manualPerspectiveChange(2, 1)])
    textBox("This space is four dimensional. (X, Y, Z, W). ")
    textBox("You couldn't push the second crate because you weren't lined up with it in the ZW plane. ")
    textBox("To move in the left perspective, use the WASD keys.", closeFunctions=[
        lambda: Character.controlled.canvas.bind("<KeyPress-w>", Character.controlled.Press),
        lambda: Character.controlled.canvas.bind("<KeyPress-a>", Character.controlled.Press),
        lambda: Character.controlled.canvas.bind("<KeyPress-s>", Character.controlled.Press),
        lambda: Character.controlled.canvas.bind("<KeyPress-d>", Character.controlled.Press),
        lambda: changeMenuText(
            "Z, X: advance, skip text. Arrow keys: Move (right). WASD: Move (left). Backspace: Rewind.")
    ])
    textBox("Try pushing one crate to the top right (8, 8, 8, 8). ... I'll de-spawn the other one.")
    textBox("...",
            closeFunctions=[
                lambda: closeBoxes(),
                lambda: clearText(),
                lambda: Goal.spawnChar(8, 8, 8, 8, 8),
                lambda: canvas.tag_raise(Crate.imageID_Right),
                lambda: canvas.tag_raise(Crate.imageID_Left),
                lambda: canvas.tag_raise(You.imageID_Right),
                lambda: canvas.tag_raise(You.imageID_Left),
                lambda: Crate2.deSpawnChar(),
                lambda: Character.winCon.append(
                    lambda: Crate.position == Goal.position
                ),
                lambda: Character.winFunc.append(runQueue),
                lambda: Character.winFunc.append(Goal.deSpawnChar),
                lambda: changeMenuText(
                    "Z, X: advance, skip text. Arrow keys: Move (right). WASD: Move (left). Backspace: Rewind. Goal: Crate"
                    " to (8, 8, 8, 8)")
            ])
    textBox("Nice! ... Sometimes the best way to solve a problem is to look at it from a different perspective.",
            closeFunctions=[
                lambda: changeMenuText(
                    "Z, X: advance, skip text. Arrow keys: Move (right). WASD: Move (left). Backspace: Rewind.")
            ])
    textBox("Believe it or not, sometimes even these two perspectives aren't enough. "
            "There are actually six independent ways to view four space.")
    textBox("To switch the left perspective, press 1. To switch the right perspective, press 2.",
            closeFunctions=[lambda: Character.controlled.canvas.bind("<KeyPress-1>", switchPerspective),
                            lambda: Character.controlled.canvas.bind("<KeyPress-2>", switchPerspective),
                            lambda: changeMenuText(
                                "Z, X: advance, skip text. Arrow keys: Move (right). WASD: Move (left). "
                                "Backspace: Rewind. 1, 2: Switch perspective")
                            ])
    textBox("Try moving the crate to (1, 1, 1, 1). I'll mess up your perspectives a bit. Start by rewinding.")
    textBox("...",
            closeFunctions=[
                lambda: closeBoxes(),
                lambda: clearText(),
                lambda: Goal.spawnChar(1, 1, 1, 1, 8),
                lambda: canvas.tag_raise(Crate.imageID_Right),
                lambda: canvas.tag_raise(Crate.imageID_Left),
                lambda: canvas.tag_raise(You.imageID_Right),
                lambda: canvas.tag_raise(You.imageID_Left),
                lambda: Crate2.deSpawnChar(),
                lambda: Character.winCon.append(
                    lambda: Crate.position == Goal.position
                ),
                lambda: manualPerspectiveChange(4, 4),
                lambda: Character.winFunc.append(runQueue),
                lambda: Character.winFunc.append(Goal.deSpawnChar),
                lambda: changeMenuText(
                    "Z, X: advance, skip text. Arrow keys: Move (right). WASD: Move (left). Backspace: "
                    "Rewind. 1, 2: Switch perspective."
                    " Goal: Crate to (1, 1, 1, 1)")
            ])
    textBox(
        "Well done. In general, it can be helpful to start puzzles by looking at two perspectives "
        "that don't share a dimension. "
        "Such pairs are called \"independent\"", closeFunctions=[lambda: changeMenuText(
            "Z, X: advance, skip text. Arrow keys: Move (right). WASD: Move (left). Backspace: "
            "Rewind. 1, 2: Switch perspective.")])
    textBox("My favorite independent perspective pair (PP) is XY ZW. "
            "If you are ever stuck on a puzzle, try looking at a different PP.")
    textBox("You may have also noticed that these perspectives are labeled as \"projections.\" ")
    textBox("A projection is like a shadow. You are looking at a 2 dimensional outline of 4 space, but there is no "
            "sense of depth.")
    textBox("To get a sense of depth you would need at least two more pieces of information. That information can "
            "be obtained by switching from a \"projection\" perspective to a \"slice\" perspective. ")
    textBox("To switch to a slice perspective, press 3 or 4.",
            closeFunctions=[gainControl,
                            lambda: changeMenuText(
                                "Z, X: advance, skip text. Arrow keys: Move (right)."
                                " WASD: Move (left). Backspace: Rewind. 1, 2: Switch perspective."
                                " 3, 4: Slice/projection")
                            ])
    textBox("An \"XY - Slice\" perspective shows you only the XY slice that you are in.")
    textBox("Suppose you are at the position (1, 2, 3, 4). An XY slice would show you any"
            " object that goes through the plane (X, Y, 3, 4)")
    textBox("This may sound complicated right now, but trust me. Things will get better...")
    textBox("Anyways, this tutorial has gone on too long. You need some drama.")
    textBox("The real game is about to begin.")
    textBox("You now know everything that you need to begin solving puzzles on your own. Good luck, and of course...")
    textBox("Welcome to Four Space.", closeFunctions=[lambda: textBox.exitFunctions.append(
        lambda: titleScreen(root, width, height))])

    canvas.focus_set()
    canvas.pack()


def titleScreen(root, width, height):
    clearWindow(root)
    canvas = Canvas(root, width=width, height=height, bg='black')
    Character.root = root
    Character.canvas = canvas
    img = PhotoImage(file="Pics/FourspaceTitle1.png")
    textBox.imageInstances.append(img)
    box = Character.canvas.create_image((width / 2, height / 2 - 100), image=img)
    Character.canvas.create_text(width / 2, 3 * height / 4 - 100, anchor="center", text="Four Space",
                                 fill="white",
                                 font="system 40")
    Character.canvas.create_text(width / 2, 3 * height / 4, anchor="center", text="by Jack Meyer Garvey",
                                 fill="white",
                                 font="system 18")
    textBox.imagesShown.append(box)
    canvas.after(4000, Billboard, root, width, height)
    changeMenuText(
        "Z, X: advance, skip text. Arrow keys: Move (right)."
        " WASD: Move (left). Backspace: Rewind. 1, 2: Switch perspective."
        " 3, 4: Slice/projection")
    canvas.pack()


def spawnPuzzle1():
    setPerspectives(1, 2)
    Crate = Character()
    Goal = Character()
    Goal.picture = PhotoImage(file="Pics/Goal.png")
    Crate.picture = PhotoImage(file="Pics/Crate.png")
    Goal.push = False
    for _ in range(3):
        for __ in range(3):
            for ___ in range(3):
                for ____ in range(3):
                    if not (_ == 1 and __ == 1 and ___ == 1):
                        Wall = Character()
                        Wall.picture = PhotoImage(file="Pics/Wall.png")
                        Wall.spawnChar(_ + 3, __ + 3, ___ + 3, ____ + 3, 8)
                        Wall.stop = True
    Goal.spawnChar(7, 7, 7, 7, 8)
    Crate.spawnChar(4, 4, 4, 4, 8)
    You = Character()
    You.picture = PhotoImage(file="Pics/Blue Square.png")
    You.spawnChar(1, 1, 1, 1, 8)
    Character.controlled = You
    textBox.exitFunctions.append(lambda: gainControl())
    Character.winCon.append(lambda: Goal.position == Crate.position)
    Character.winFunc.append(
        lambda: textBox("You win!!!!!!", closeFunctions=[
            lambda: textBox.exitFunctions.append(lambda: Puzzle2(textBox.root, 1024, 768))]))
    Character.winFunc.append(lambda: runQueue())


def Puzzle1(root, width, height):
    clearWindow(root)
    canvas = Canvas(root, width=width, height=height, bg='black')
    Character.root = root
    Character.canvas = canvas
    textBox.canvas = canvas
    textBox.root = root
    text = Character.canvas.create_text(width / 2, height / 2, anchor="center", text="Puzzle 1: You are the Knife",
                                        fill="white",
                                        font="system 40")
    root.after(2000, canvas.delete, text)
    root.after(2000, runQueue)
    root.after(2000, tkinterDrawings.drawGrids, canvas, 8)
    root.after(2000, spawnPuzzle1)
    textBox("Welcome to Puzzle 1. This puzzle will probably require looking at a slice perspective.")
    textBox("The blue squares are walls.")
    textBox("Push the crate to (7, 7, 7, 7).")

    canvas.focus_set()
    canvas.pack()


def spawnPuzzle2():
    setPerspectives(1, 2)
    Crate = Character()
    Goal = Character()
    Goal.picture = PhotoImage(file="Pics/Goal.png")
    Crate.picture = PhotoImage(file="Pics/Crate.png")
    Goal.push = False
    for x in range(4):
        for y in range(4):
            for z in range(4):
                for w in range(4):
                    if not (0 < x < 3 and 0 < y < 3 and 0 < z < 3 and 0 < w):
                        Wall = Character()
                        Wall.picture = PhotoImage(file="Pics/Wall.png")
                        Wall.spawnChar(x + 3, y + 3, z + 3, w + 3, 8)

                        Wall.stop = True
    Goal.spawnChar(4, 4, 4, 4, 8)
    Crate.spawnChar(2, 2, 2, 2, 8)
    You = Character()
    You.picture = PhotoImage(file="Pics/Blue Square.png")
    You.spawnChar(1, 1, 1, 1, 8)
    Character.controlled = You
    Character.winCon.append(lambda: Goal.position == Crate.position)
    textBox.exitFunctions.append(lambda: gainControl())
    Character.winFunc.append(
        lambda: textBox("You win!!!!!!", closeFunctions=[
            lambda: textBox.exitFunctions.append(lambda: Puzzle2(textBox.root, 1024, 768))]))
    Character.winFunc.append(lambda: runQueue())


def Puzzle2(root, width, height):
    clearWindow(root)
    canvas = Canvas(root, width=width, height=height, bg='black')
    Character.root = root
    Character.canvas = canvas
    textBox.canvas = canvas
    textBox.root = root
    text = Character.canvas.create_text(width / 2, height / 2, anchor="center", text="Puzzle 2: Four Bucket",
                                        fill="white",
                                        font="system 40")
    root.after(2000, canvas.delete, text)
    root.after(2000, runQueue)
    root.after(2000, tkinterDrawings.drawGrids, canvas, 8)
    root.after(2000, spawnPuzzle2)
    textBox("Welcome to Puzzle 2.")
    textBox("Push the crate to (4, 4, 4, 4).")

    canvas.focus_set()
    canvas.pack()


def startLevel(root, width, height, n):
    clearWindow(root)
    canvas = Canvas(root, width=width, height=height, bg='black')
    Character.root = root
    Character.canvas = canvas
    textBox.canvas = canvas
    textBox.root = root
    tkinterDrawings.drawGrids(canvas, n, grid=False)
    setPerspectives(1, 1)
    setSlice(0, 0)
    canvas.focus_set()
    canvas.pack()


def spawnYou(n, x=1, y=1, z=1, w=1):
    You = Character()
    Character.controlled = You
    You.picture = PhotoImage(file="Pics/Blue Square.png")
    You.spawnChar(x, y, z, w, n)


def Billboard(root, width, height, x=2, y=2, z=1, w=1):
    n = 16
    startLevel(root, width, height, n)
    arrow = Character()
    arrow.picture = PhotoImage(file="Pics/Arrow.png")
    arrow.push = False
    arrow.spawnChar(15, 2, 1, 1, 16)
    arrow = Character()
    arrow.picture = PhotoImage(file="Pics/ArrowUp.png")
    arrow.push = False
    arrow.spawnChar(2, 15, 1, 1, 16)
    if "inspectScreen" in savedState:
        spawnBox(2, 6, 1, 1, n, pic="Pics/paper.png",
                 inspec=[lambda:
                         textBox(
                             "Note: some rooms appear to be in 2D. If the room loads in as a slice, it is probably 2D.",
                             "Pics/GuyFaceBox3.png"),
                         lambda: textBox("P.S. If you are reading this, you failed the puzzle.",
                                         "Pics/GuyFaceBox.png"),
                         lambda: textBox("P.P.S. So did I", "Pics/GuyFaceBox2.png")
                         ])
    spawnYou(n, x=x, y=y, z=z, w=w)
    for _ in range(n):
        if _ > 3 and _ not in [5, 6, 7, 8, 9]:
            spawnBox(13, _ + 1, 1, 1, n)
        if _ in [5, 9]:
            spawnBox(12, _ + 1, 1, 1, n)
        if _ in [6]:
            spawnBox(10, _ + 1, 1, 1, n)
    for _ in range(3):
        spawnBox(13, _ + 7, 1, 1, n, stop=False, push=True, pic="Pics/WhiteBoxDot.png")
    for _ in range(12):
        spawnBox(_ + 1, 8, 1, 1, n)
    for _ in range(n):
        Box = Character()
        Box.picture = PhotoImage(file="Pics/WhiteBox.png")
        Box.stop = True
        Box.spawnChar(_ + 1, 4, 1, 1, n)
    for _ in range(3):
        crate = Character()
        crate.picture = PhotoImage(file="Pics/WhiteBoxDot.png")
        crate.spawnChar(11, _ + 1, 1, 1, 16)
    Character.winCon.append(
        lambda: Character.controlled.position[0] == 16 and Character.controlled.looking == (1, 0, 0, 0))
    Character.winFunc.append(lambda: middleRoom(root, width, height, y=Character.controlled.position[1]))
    setSlice(1, 1)
    gainControl(stuck=True)
    Character.controlled.canvas.bind("<KeyPress-BackSpace>", lambda event: rewind())
    Character.flags.append((lambda: Character.controlled.position in [[1, 16, 1, 1], [2, 16, 1, 1], [3, 16, 1,
                                                                                                     1]] and Character.controlled.looking == (
                                        0, 1, 0, 0),
                            lambda: keyPuzzle(root, width, height, x=Character.controlled.position[0], y=1)))
    generateSlice(("000000111100",
                   "003001020000",
                   "000010220000",
                   "001102201000",
                   "000022001000",
                   "001102201000",
                   "000111111111",
                   "000100000000"), (1, 1, 0, 0), (1, 9, 1, 1), n)


def spawnBox(x, y, z, w, n, pic="Pics/WhiteBox.png", push=True, stop=True, slide=False, inspec=None):
    if inspec is not None:
        Box = Character(inspec)
    else:
        Box = Character()
    Box.picture = PhotoImage(file=pic)
    Box.push = push
    Box.stop = stop
    Box.slide = slide
    Box.spawnChar(x, y, z, w, n)


def inspectScreen(root, width, height, x=1, y=2, z=1, w=1):
    n = 16
    startLevel(root, width, height, n)
    Character.winCon.append(
        lambda: Character.controlled.position[0] == 1 and Character.controlled.looking == (-1, 0, 0, 0))
    Character.winFunc.append(lambda: firstKeyPuzzle(root, width, height, x=16, y=Character.controlled.position[1]))
    for _ in range(3):
        spawnBox(4, _ + 4, 1, 1, n)
        spawnBox(8, _ + 4, 1, 1, n)

    spawnBox(5, 6, 1, 1, n)
    spawnBox(7, 6, 1, 1, n)
    spawnBox(6, 7, 1, 1, n)
    spawnBox(5, 7, 1, 1, n)
    spawnBox(7, 7, 1, 1, n)

    spawnBox(2, 9, 1, 1, n, "Pics/ArrowLeft.png", False, False)
    for _ in range(n):
        if _ not in [4, 5, 6, 7, 12, 13, 14, 15]:
            spawnBox(_ + 1, 4, 1, 1, n)
    spawnBox(12, 5, 1, 1, n)
    spawnBox(12, 6, 1, 1, n)
    spawnBox(12, 7, 1, 1, n)
    spawnBox(12, 8, 1, 1, n)
    spawnBox(12, 9, 1, 1, n)
    spawnBox(15, 9, 1, 1, n)
    spawnBox(13, 9, 1, 1, n)
    spawnBox(14, 11, 1, 1, n)
    spawnBox(16, 9, 1, 1, n)
    if "inspectScreen" not in savedState:
        Box = Character()
        Box.picture = PhotoImage(file="Pics/WhiteBoxDot.png")
        Box.push = True
        Box.spawnChar(14, 10, 1, 1, n)
        Guy = Character()
        Guy.picture = PhotoImage(file="Pics/Chair Guy1.png")
        Guy.frames.append(PhotoImage(file="Pics/Chair Guy1.png"))
        Guy.frames.append(PhotoImage(file="Pics/Chair Guy2.png"))
        Guy.spawnChar(6, 10, 1, 1, n)
        Guy.animate(500)
        icey = Character()
        icey.picture = PhotoImage(file="Pics/Icey.png")

        Blank = Character(inspectFunctions=[lambda: textBox(
            "You know how to talk to me? ... I didn't even tell you how.",
            face="Pics/GuyFaceBox1.png"), lambda: textBox(
            "You saw the tutorial too right?",
            face="Pics/GuyFaceBox.png"), lambda: textBox(
            "You skipped it?", "Pics/GuyFaceBox3.png"), lambda: textBox(
            "Then you must know this place is four dimensional. ",
            face="Pics/GuyFaceBox1.png"),
                                            lambda: textBox(
                                                "I'm here doing research myself actually. I'm gonna beat this game faster than anyone!",
                                                face="Pics/GuyFaceBox1.png"),
                                            lambda: textBox(
                                                "Want to know a secret? I found a mechanic the tutorial never mentioned.",
                                                "Pics/GuyFaceBox3.png"),
                                            lambda: textBox(
                                                "If you move towards an object and press Z, sometimes, there will be a message. Pretty cool right?",
                                                face="Pics/GuyFaceBox.png"), lambda: textBox(
                "That won't be very helpful for my speedrun. I'll probably just ignore all the flavor text.",
                face="Pics/GuyFaceBox.png"), lambda: textBox(
                "Its unfortunate that this game is filled with jokes, because I hate cutscenes.",
                "Pics/GuyFaceBox3.png"),
                                            lambda: textBox(
                                                "Ugh, I think a joke is about to start.",
                                                face="Pics/GuyFaceBox2.png", closeFunctions=[
                                                    lambda: icey.spawnChar(15, 10, 1, 1, n),
                                                    lambda: root.after(700, icey.manualPress, -1, 0, 0, 0),
                                                    lambda: root.after(1400, icey.manualPress, -1, 0, 0, 0)]),
                                            lambda: textBox("... ...", "Pics/GuyFaceBox2.png"),
                                            lambda: textBox("Oh boy, I sure love moving in the X direction!",
                                                            "Pics/Icey.png"),
                                            lambda: textBox("... I mean, what the hell even is that?",
                                                            "Pics/GuyFaceBox3.png"),
                                            lambda: textBox("Oh ... uh ... I'll see you up ahead!",
                                                            "Pics/GuyFaceBox.png",
                                                            closeFunctions=[
                                                                lambda: icey.manualPress(-1, 0, 0, 0),
                                                                lambda: root.after(200, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(400, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(600, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(800, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(1000, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(1200, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(1400, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(1600, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(1800, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(1800, Guy.deSpawnChar),
                                                                lambda: root.after(2000, icey.manualPress, -1, 0, 0, 0),
                                                                lambda: root.after(2200, icey.deSpawnChar),
                                                                lambda: root.after(1800, Box.deSpawnChar)
                                                            ]),
                                            lambda: Blank.deSpawnChar(),
                                            lambda: addSavedState("inspectScreen", "Done")
                                            ])
        if savedState["tutorial"] == "Played":
            Blank.inspectFunctions.pop(2)
        Blank.push = False
        Blank.picture = PhotoImage(file="Pics/Blank.png")
        Blank.spawnChar(6, 7, 1, 1, n)
        Character.flags.append((lambda: Character.controlled.position[0] == 11 and "inspectScreen" not in savedState,
                                lambda: loseControl(Character.canvas),
                                lambda: textBox("Hey! Whats the matter with you?",
                                                exitFunctions=[lambda: gainControl(stuck=True)],
                                                face="Pics/GuyFaceBox.png"),
                                lambda: textBox("Don't you know how to greet a new pal?",
                                                face="Pics/GuyFaceBox.png"),
                                lambda: textBox("Walk up to me and press z.", face="Pics/GuyFaceBox.png"),
                                lambda: runQueue(), lambda: Blank.inspectFunctions.pop(0)))

    spawnYou(n, x=x, y=y, z=z, w=w)
    setSlice(1, 1)
    gainControl(stuck=True)


def generateSlice(array, direction, anchor, n):
    addition = [0, 0, 0, 0]
    for _ in array:
        for __ in range(len(direction)):
            if direction[__] != 0:
                addition[__] = 0
                break
        for __ in _:
            if __ == "0":
                pass
            elif __ == "1":
                spawnBox(anchor[0] + addition[0], anchor[1] + addition[1], anchor[2] + addition[2],
                         anchor[3] + addition[3], n)
            elif __ == "2":
                spawnBox(anchor[0] + addition[0], anchor[1] + addition[1], anchor[2] + addition[2],
                         anchor[3] + addition[3],
                         n, stop=False, push=True, pic="Pics/WhiteBoxDot.png")
            elif __ == "3":
                spawnBox(anchor[0] + addition[0], anchor[1] + addition[1], anchor[2] + addition[2],
                         anchor[3] + addition[3],
                         n, stop=False, push=True, slide=True, pic="Pics/SlideBox.png")
            for ___ in range(len(direction)):
                if direction[___] != 0:
                    addition[___] += 1
                    break
        skip = True
        for __ in range(len(direction)):
            if direction[__] != 0:
                if skip is False:
                    addition[__] += 1
                else:
                    skip = False


def keyPuzzle(root, width, height, x=1, y=1, z=1, w=1):
    n = 16
    startLevel(root, width, height, n)
    setSlice(1, 1)
    spawnBox(14, 15, 1, 1, n, pic="Pics/Icey.png", push=False,
             inspec=[lambda: textBox("..........", "Pics/Icey.png")])
    spawnBox(16, 15, 1, 1, n, pic="Pics/Paper.png", push=False,
             inspec=[
                 lambda: textBox("Note: These light blue characters are very helpful. They give hints to cool people.",
                                 "Pics/GuyFaceBox3.png")])
    spawnBox(15, 6, 1, 1, n, pic="Pics/Arrow.png", stop=False, push=False)
    spawnYou(n, x, y, z, w)

    Character.flags.append((lambda: Character.controlled.position in [[1, 1, 1, 1], [2, 1, 1, 1],
                                                                      [3, 1, 1,
                                                                       1]] and Character.controlled.looking == (
                                        0, -1, 0, 0),
                            lambda: Billboard(root, width, height, x=Character.controlled.position[0], y=16)))
    Character.flags.append((lambda: Character.controlled.position in [[16, 5, 1, 1], [16, 6, 1, 1],
                                                                      [16, 7, 1,
                                                                       1]] and Character.controlled.looking == (
                                        1, 0, 0, 0),
                            lambda: smallRoom(root, width, height, y=Character.controlled.position[1])))
    gainControl(stuck=True)
    generateSlice(("0001000000001000",
                   "0001011111101000",
                   "0003000000001000",
                   "0001110111101111",
                   "0001010020100000",
                   "0001110100110000",
                   "3330110100100000",
                   "3330113103111111",
                   "0001000000300320",
                   "0002010010000300",
                   "0111020220000101",
                   "0001113113111101",
                   "1001000010220000",
                   "0001022030110000",
                   "0001010110100000",
                   "0000010000100000"), (1, 1, 0, 0), (1, 1, 1, 1), n)


def fillSlice(direction, anchor, n):
    """Fills in the n-by-n slice specified in 4 space with blocks"""
    array = []
    for _ in range(n):
        row = ""
        for __ in range(n):
            row += "1"
        array.append(row)
    generateSlice(tuple(array), direction, anchor, n)
    print("Done")


def smallRoom(root, width, height, x=1, y=1, z=1, w=1):
    n = 16
    startLevel(root, width, height, n)
    Character.flags.append((lambda: Character.controlled.position in [[1, 5, 1, 1], [1, 6, 1, 1], [1, 7, 1, 1]] and
                                    Character.controlled.looking == (-1, 0, 0, 0),
                            lambda: keyPuzzle(root, width, height, x=16, y=Character.controlled.position[1])))
    spawnBox(4, 4, 4, 4, n, pic="Pics/StandingGuy.png", push=False, stop=True,
             inspec=[lambda: textBox("Yo", "Pics/GuyFacebox.png")])
    generateSlice(("0",
                   "0",
                   "0",
                   "1",
                   "0",
                   "0",
                   "0",
                   "1",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0"), (1, 1, 0, 0), (1, 1, 1, 1), n)
    for _ in range(3):
        for __ in range(3):
            for ___ in range(3):
                for ____ in range(3):
                    if not (_ == 1 and __ == 1 and ___ == 1):
                        spawnBox(_ + 6, __ + 6, ___ + 7, ____ + 8, n)
                    elif ____ == 1:
                        spawnBox(_ + 6, __ + 6, ___ + 7, ____ + 8, n, stop=False, push=True, pic="Pics/Icey.png")
    spawnYou(n, x, y, z, w)
    setSlice(1, 0)
    manualPerspectiveChange(1, 2)
    gainControl()


def middleRoom(root, width, height, x=1, y=1, z=1, w=1):
    n = 16
    startLevel(root, width, height, n)
    Character.flags.append((lambda: Character.controlled.position[0] == 1 and
                                    Character.controlled.looking == (-1, 0, 0, 0),
                            lambda: Billboard(root, width, height, x=16, y=Character.controlled.position[1])))
    Character.flags.append((lambda: Character.controlled.position[0] == 16 and
                                    Character.controlled.looking == (1, 0, 0, 0),
                            lambda: firstKeyPuzzle(root, width, height, x=1, y=Character.controlled.position[1])))
    generateSlice(("000000003000",
                   "000000003000",
                   "000000003000",
                   "1111111111111111",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0",
                   "0"), (1, 1, 0, 0), (1, 1, 1, 1), n)
    spawnBox(15, 2, 1, 1, n, pic="Pics/Arrow.png", stop=False, push=False)
    spawnYou(n, x, y, z, w)
    setSlice(1, 1)
    gainControl(stuck=True)


def firstKeyPuzzle(root, width, height, x=1, y=1, z=1, w=1):
    n = 16
    startLevel(root, width, height, n)
    Character.flags.append((lambda: Character.controlled.position[0] == 1 and
                                    Character.controlled.looking == (-1, 0, 0, 0),
                            lambda: middleRoom(root, width, height, x=16, y=Character.controlled.position[1])))
    Character.flags.append((lambda: Character.controlled.position[0] == 16 and
                                    Character.controlled.looking == (1, 0, 0, 0),
                            lambda: inspectScreen(root, width, height, x=1, y=Character.controlled.position[1])))
    generateSlice(("000030000000001",
                   "0000112101101",
                   "0000000100001",
                   "1111111111111111",
                   "0000000100000000",
                   "0000000100000000",
                   "0000000100000000",
                   "0000000100000000",
                   "0000000100000000",
                   "000000010000000",
                   "0000000300000000",
                   "0000000100000000",
                   "0000000100000000",
                   "0000000100000000",
                   "0000000100000000",
                   "0000000100000000"), (1, 1, 0, 0), (1, 1, 1, 1), n)
    spawnBox(15, 2, 1, 1, n, pic="Pics/Arrow.png", stop=False, push=False)
    spawnYou(n, x, y, z, w)
    setSlice(1, 1)
    gainControl(stuck=True)
