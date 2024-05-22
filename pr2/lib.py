import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from main import *
from pqdict import pqdict

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


def weightedAStarHeuristic(point, goal, epsilon):
    return epsilon * np.linalg.norm(point - goal)


def searchBasedPlan(start, goal, mapDirectory, GRID_UPSCALE, horizon, epsilon):
    CLOSE_THRESHOLD = 2
    boundary, blocks = load_map(mapDirectory)

    meshX = np.arange(boundary[0,0] * GRID_UPSCALE, (boundary[0,3] * GRID_UPSCALE) + 1)
    meshY = np.arange(boundary[0,1] * GRID_UPSCALE, (boundary[0,4] * GRID_UPSCALE) + 1)
    meshZ = np.arange(boundary[0,2] * GRID_UPSCALE, (boundary[0,5] * GRID_UPSCALE) + 1)

    meshX, meshY, meshZ = np.meshgrid(meshX, meshY, meshZ)

    mesh = np.vstack((meshX.flatten(),meshY.flatten(),meshZ.flatten())).T

    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    #directionsPossible = [np.array([1,0,0]), np.array([0,-1,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([0,0,-1]), np.array([1,1,0]), np.array([-1,-1,0]), np.array([1,-1,0]), np.array([-1,1,-0]), np.array([1,1,1]), np.array([-1,-1,1]), np.array([1,-1,1]), np.array([-1,1,1]), np.array([1,1,-1]), np.array([-1,-1,-1]), np.array([1,-1,-1]), np.array([-1,1,-1])]

    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    #print(dR)
    dR = np.delete(dR,13,axis=1)
    directionsPossible = dR.T

    validMeshPoints = []
    invalidMeshPoints = []

    totalMeshPoints = mesh.shape[0]
    i = 0

    for meshPoint in mesh:
            if not checkCollisionPointAABB(meshPoint / GRID_UPSCALE, blocks) and checkCollisionPointAABB(meshPoint / GRID_UPSCALE, boundary):
                validMeshPoints.append(meshPoint)
            else:
                invalidMeshPoints.append(meshPoint)
            #print(i / totalMeshPoints)
            i = i + 1

    validMeshPoints = np.array(validMeshPoints)
    invalidMeshPoints = np.array(invalidMeshPoints)

    #ax.scatter(validMeshPoints[:,0] / GRID_UPSCALE, validMeshPoints[:,1] / GRID_UPSCALE, validMeshPoints[:,2] / GRID_UPSCALE, s=1, c='orange')


    totalMeshPoints = validMeshPoints.shape[0]
    i = 0


    startCoordinate = start * GRID_UPSCALE
    goalCoordinate = goal * GRID_UPSCALE

    graph = {}

    for meshPoint in validMeshPoints.astype(np.int16):
            graph[tuple(meshPoint)] = Node(tuple(meshPoint), np.inf, weightedAStarHeuristic(meshPoint, goalCoordinate, epsilon), [])

    for node in graph.values():
            print(i / totalMeshPoints)
            i = i + 1
            for directionPossible in directionsPossible:
                if tuple(node.label + directionPossible.astype(np.int16)) in graph:
                    collisionFree = True
                    for block in blocks:
                        if checkCollision(np.array(node.label) / GRID_UPSCALE, (np.array(node.label) + directionPossible.astype(np.int16)) / GRID_UPSCALE, block):
                            collisionFree = False
                        if not collisionFree:
                            break
                    if collisionFree:
                        cost = np.linalg.norm(directionPossible)
                        node.childrenAndCosts.append((tuple(node.label + directionPossible.astype(np.int16)), cost))




    START_NODE = tuple(startCoordinate)
    GOAL_NODE = tuple(goalCoordinate)




    if START_NODE not in graph:
        graph[START_NODE] = Node(START_NODE, np.inf, weightedAStarHeuristic(start, goalCoordinate, epsilon), [])
        distanceToOtherNodes = np.linalg.norm(validMeshPoints - startCoordinate, axis=1)
        indexNodesClose = np.where(distanceToOtherNodes < CLOSE_THRESHOLD)[0]
        for index in indexNodesClose:
            nearbyCoordinate = validMeshPoints[index,:]
            collisionFree = True
            for block in blocks:
                if checkCollision(startCoordinate / GRID_UPSCALE, nearbyCoordinate / GRID_UPSCALE, block):
                    collisionFree = False
                if not collisionFree:
                        break
            if collisionFree:
                cost = np.linalg.norm(startCoordinate - nearbyCoordinate)
                graph[START_NODE].childrenAndCosts.append((tuple(nearbyCoordinate), cost))

    if GOAL_NODE not in graph:
        graph[GOAL_NODE] = Node(GOAL_NODE, np.inf, 0, [])
        distanceToOtherNodes = np.linalg.norm(validMeshPoints - goalCoordinate, axis=1)
        indexNodesClose = np.where(distanceToOtherNodes < CLOSE_THRESHOLD)[0]
        for index in indexNodesClose:
            nearbyCoordinate = validMeshPoints[index,:]
            collisionFree = True
            for block in blocks:
                if checkCollision(goalCoordinate / GRID_UPSCALE, nearbyCoordinate / GRID_UPSCALE, block):
                    collisionFree = False
                if not collisionFree:
                        break
            if collisionFree:
                cost = np.linalg.norm(startCoordinate - nearbyCoordinate)
                graph[tuple(nearbyCoordinate)].childrenAndCosts.append((GOAL_NODE, cost))

    open = pqdict()
    closed = []

    graph[START_NODE].g = 0

    open[START_NODE] = graph[START_NODE].g + graph[START_NODE].h

    i = 0

    while GOAL_NODE not in closed:

        if i == horizon:
             return 0, False

        print(i)

        currentNode = graph[open.pop()]
        #print("current node: " + str(currentNode.label))

        closed.append(currentNode.label)
        for (child, cost) in currentNode.childrenAndCosts:
            if child not in closed:
                if graph[child].g > currentNode.g + cost:
                    graph[child].g = currentNode.g + cost
                    graph[child].parent = currentNode.label
                    open[child] = graph[child].g + graph[child].h
        
        #print("open: " + str(open))
        #print("closed: " + str(closed))

        i = i + 1

    shortestPath = [GOAL_NODE]

    currentTraceNode = GOAL_NODE

    totalCost = 0

    while currentTraceNode != START_NODE:
        cost = np.linalg.norm(np.array(graph[currentTraceNode].label) - np.array(graph[currentTraceNode].parent))
        totalCost = totalCost + cost
        currentTraceNode = graph[currentTraceNode].parent
        shortestPath.append(currentTraceNode)

    shortestPath.reverse()
    shortestPath = np.array(shortestPath)

    ax.plot(shortestPath[:,0] / GRID_UPSCALE,shortestPath[:,1] / GRID_UPSCALE,shortestPath[:,2] / GRID_UPSCALE,'r-')
                    
    #plt.show(block=True)

    return totalCost / GRID_UPSCALE, True
