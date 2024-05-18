import numpy as np

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

    tx_low = 0
    ty_low = 0
    tz_low = 0
    tx_high = 1
    ty_high = 1
    tz_high = 1

    if rx != 0:
        tx_low = (lx - point1[0]) / rx
        tx_high = (hx - point1[0]) / rx
        

    if ry != 0:
        ty_low = (ly - point1[1]) / ry
        ty_high = (hy - point1[1]) / ry

    if rz != 0:
        tz_low = (lz - point1[2]) / rz
        tz_high = (hz - point1[2]) / rz

    tx_close = np.min(np.array([tx_low, tx_high]))
    tx_far = np.max(np.array([tx_low, tx_high]))
    ty_close = np.min(np.array([ty_low, ty_high]))
    ty_far = np.max(np.array([ty_low, ty_high]))
    tz_close = np.min(np.array([tz_low, tz_high]))
    tz_far = np.max(np.array([tz_low, tz_high]))

    t_close = np.max(np.array([tx_close, ty_close, tz_close]))
    t_far = np.min(np.array([tx_far, ty_far, tz_far]))

    if (t_close > 1 or t_close < 0) and (t_far > 1 or t_far < 0) and (t_close * t_far > 0):
        return False
    else:
        return t_close <= t_far


