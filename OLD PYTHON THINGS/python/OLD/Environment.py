from tkinter import *
from python.OLD.GameObject import GameObject

root = Tk()

root.geometry("800x500")
root.title("NAISENT!")

space = Canvas(root, width=800, height=500, bg="black", borderwidth=0, highlightthickness=0)
space.pack()

rightDirection = 0
leftDirection = 0

def bind_right(_event):
    global rightDirection
    rightDirection = 1
def bind_left(_event):
    global leftDirection
    leftDirection = -1
    
def unbind_right(_event):
    global rightDirection
    rightDirection = 0
def unbind_left(_event):
    global leftDirection
    leftDirection = 0

root.bind("<d>", bind_right)
root.bind("<a>", bind_left)
root.bind("<KeyRelease-d>", unbind_right)
root.bind("<KeyRelease-a>", unbind_left)

ball = GameObject.circle([400, 100], 30)
ball.display(root, color="blue")
pad = GameObject.rectangle([400,400], [100, 20])
pad.display(root, color="white")

ballCollidingDebounce = False
def removeCollidingDebounce():
    global ballCollidingDebounce
    ballCollidingDebounce = False

def update():
    global ballCollidingDebounce
    if pad.isColliding(ball) & (not ballCollidingDebounce):
        print("wow")

        xDifference = ball.position[0]-pad.position[0]
        if not xDifference == 0:
            xDifference = xDifference/abs(xDifference)
        ball.velocity = [-(xDifference)*.2, 1.2]

        ballCollidingDebounce = True
        root.after(200, removeCollidingDebounce)

    if ball.position[0]+ball.size[0]/2 >= 800:
        ball.velocity = [.1, ball.velocity[1]]
    elif ball.position[0]-ball.size[0]/2 <= 0:
        ball.velocity = [-.1, ball.velocity[1]]

    ball.velocityUpdate()
    pad.move((3/10)*(rightDirection+leftDirection), 0)
    if pad.position[0] > 800:
        pad.setPosition(800, pad.position[1])
    elif pad.position[0] < 0:
        pad.setPosition(0, pad.position[1])

    root.after(1, update)

root.after(1, update)

root.mainloop()

