def checkCollision(point1, point2, block):
    lx = block[0]
    ly = block[1]
    lz = block[2]
    hx = block[0]
    hy = block[1]
    hz = block[2]

    lineDirection = point1 - point2
    rx = lineDirection[0]
    ry = lineDirection[1]
    rz = lineDirection[2]
    
