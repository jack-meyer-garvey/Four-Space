def makeGrid(canvas, n, zero_zero, top_top, grid=True):
    """Creates an n by n grid"""
    canvas.create_rectangle(zero_zero[0], zero_zero[1], top_top[0], top_top[1], outline='white', width=4)
    if grid is True:
        for _ in range(n - 1):
            canvas.create_line(zero_zero[0], top_top[1] + (_ + 1) * (zero_zero[1] - top_top[1]) / n, top_top[0],
                               top_top[1] + (_ + 1) * (zero_zero[1] - top_top[1]) / n, fill='white', width=1)
            canvas.create_line(top_top[0] - (_ + 1) * (top_top[0] - zero_zero[0]) / n, zero_zero[1],
                               top_top[0] - (_ + 1) * (top_top[0] - zero_zero[0]) / n, top_top[1], fill='white',
                               width=1)


def grid_to_screen(x, y, n, side):
    """Coverts an n by n grid position (x, y) into the screen pixel position"""
    if side == "Right":
        top_top = [1008, 24.0]
        zero_zero = [528, 504.0]
    if side == "Left":
        top_top = [496, 24.0]
        zero_zero = [16, 504.0]
    xpos = zero_zero[0] + (x - 0.5) * (top_top[0] - zero_zero[0]) / n
    ypos = zero_zero[1] + (y - 0.5) * (top_top[1] - zero_zero[1]) / n
    return [xpos, ypos]


def drawGrids(canvas, n, grid=True):
    makeGrid(canvas, n, (496, 24.0), (16, 504.0), grid)
    makeGrid(canvas, n, (1008, 24.0), (528, 504.0), grid)
