import numpy as np



class Node():
    def __init__(self, label, g, h, childrenAndCosts):
        self.label = label
        self.g = g
        self.h = h
        self.parent = None
        self.childrenAndCosts = childrenAndCosts
    def print(self):
        print("--------------------------------------")
        print("label: " + str(self.label))
        print("g: " + str(self.g))
        print("h: " + str(self.h))
        print("parent: " + str(self.parent))
        print("(children,cost): " + str(self.childrenAndCosts))
        print("--------------------------------------")

def checkCollisionPointAABB(point, blocks):
    collision = False
    for block in blocks:
        if point[0] > block[0] and point[0] < block[3] and\
            point[1] > block[1] and point[1] < block[4] and\
            point[2] > block[2] and point[2] < block[5]:
                collision = True
                break
    return collision
    

def checkCollision(point1, point2, block):

    lx = block[0]
    ly = block[1]
    lz = block[2]
    hx = block[3]
    hy = block[4]
    hz = block[5]

    lineDirection = point2 - point1
    rx = lineDirection[0]
    ry = lineDirection[1]
    rz = lineDirection[2]


    tx_low = (lx - point1[0]) / rx
    tx_high = (hx - point1[0]) / rx
    
    ty_low = (ly - point1[1]) / ry
    ty_high = (hy - point1[1]) / ry

    tz_low = (lz - point1[2]) / rz
    tz_high = (hz - point1[2]) / rz


    tx_close = np.nanmin(np.array([tx_low, tx_high]))
    tx_far = np.nanmax(np.array([tx_low, tx_high]))
    ty_close = np.nanmin(np.array([ty_low, ty_high]))
    ty_far = np.nanmax(np.array([ty_low, ty_high]))
    tz_close = np.nanmin(np.array([tz_low, tz_high]))
    tz_far = np.nanmax(np.array([tz_low, tz_high]))

    t_close = np.nanmax(np.array([tx_close, ty_close, tz_close]))
    t_far = np.nanmin(np.array([tx_far, ty_far, tz_far]))


    if (0 > t_close and 0 > t_far) or (1 < t_close and 1 < t_far):
        return False
    else:
        return t_close <= t_far
