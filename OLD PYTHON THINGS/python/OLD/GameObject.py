from __future__ import annotations


from tkinter import *

GRAVITY = .5

VELOCITY_NUDGE_FACTOR = .002

def nudgeTo(number1, number2, nudgeFactor):
    if number1 > number2:
        newNumber = number1-nudgeFactor
        if newNumber < number2:
            return number2
        return newNumber
    elif number1 < number2:
        newNumber = number1+nudgeFactor
        if newNumber > number2:
            return number2
        return newNumber
    else:
        return number1

def create_circle(canvas: Canvas, x, y, r, **kwargs):
    """
    A function to create a circle on a canvas.

    :param canvas: The Tkinter Canvas object.
    :param x: The x-coordinate of the center.
    :param y: The y-coordinate of the center.
    :param r: The radius of the circle.
    :param kwargs: Additional options like fill, outline, width, etc.
    """
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvas.create_oval(x0, y0, x1, y1, **kwargs)

def make_object_display(root: Tk, position, size):
    object = Canvas(root, width=size[0], height=size[1], bg="black", highlightthickness=0)
    object.pack()
    object.place(x=position[0]-size[0]/2, y=position[1]-size[1]/2)
    return object

class GameObject:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.velocity = [0,-GRAVITY]

    @staticmethod
    def circle(position, radius):
        gameObject = GameObject(position, [radius, radius])
        gameObject.type = "circle"
        return gameObject
    @staticmethod
    def square(position, size):
        gameObject = GameObject(position, [size, size])
        gameObject.type = "square"
        return gameObject
    @staticmethod
    def rectangle(position, size):
        gameObject = GameObject(position, size)
        gameObject.type = "rectangle"
        return gameObject
    
    def setPosition(self, x, y):
        self.position = [x, y]
        if self.objectDisplay:
            self.objectDisplay.place(x=self.position[0]-self.size[0]/2, y=self.position[1]-self.size[1]/2)

    def move(self, x, y):
        self.setPosition(self.position[0]+x, self.position[1]+y)

    def velocityUpdate(self):
        self.move(-self.velocity[0], -self.velocity[1])

        self.velocity = [nudgeTo(self.velocity[0], 0, VELOCITY_NUDGE_FACTOR*.05),
                         nudgeTo(self.velocity[1], -GRAVITY, VELOCITY_NUDGE_FACTOR)]

    def display(self, root: Tk, color):
        self.objectDisplay = make_object_display(root, self.position, self.size)
        if self.type == "circle":
            create_circle(self.objectDisplay, self.size[0]/2, self.size[0]/2, self.size[0]/2, fill=color)
        elif self.type == "square":
            self.objectDisplay.create_rectangle(0,0, self.size[0], self.size[1], fill=color)
        elif self.type == "rectangle":
            self.objectDisplay.create_rectangle(0,0, self.size[0], self.size[1], fill=color)

    def isColliding(self, other: GameObject):
        xIsColliding = ((self.position[0]+self.size[0]/2) >= (other.position[0]-other.size[0]/2))\
            & ((self.position[0]-self.size[0]/2) <= (other.position[0]+other.size[0]/2))
        yIsColliding = ((self.position[1]+self.size[1]/2) >= (other.position[1]-other.size[1]/2))\
            & ((self.position[1]-self.size[1]/2) <= (other.position[1]+other.size[1]/2))
        
        return xIsColliding & yIsColliding
