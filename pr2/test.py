import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from main import *
from lib import *
from pqdict import pqdict

GRID_UPSCALE = 1.5 #meters
DECIMAL_PLACE_ROUND = 3
CLOSE_THRESHOLD = 5

boundary, blocks = load_map('./maps/monza.txt')

meshX = np.arange(boundary[0,0] * GRID_UPSCALE, (boundary[0,3] * GRID_UPSCALE) + 1)
meshY = np.arange(boundary[0,1] * GRID_UPSCALE, (boundary[0,4] * GRID_UPSCALE) + 1)
meshZ = np.arange(boundary[0,2] * GRID_UPSCALE, (boundary[0,5] * GRID_UPSCALE) + 1)

meshX, meshY, meshZ = np.meshgrid(meshX, meshY, meshZ)

mesh = np.vstack((meshX.flatten(),meshY.flatten(),meshZ.flatten())).T


start = np.array([1.0, 5.0, 1.5])
goal = np.array([9.0, 7.0, 1.5])
fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)


#directionsPossible = [np.array([1,0,0]), np.array([0,-1,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([0,0,-1]), np.array([1,1,0]), np.array([-1,-1,0]), np.array([1,-1,0]), np.array([-1,1,-0]), np.array([1,1,1]), np.array([-1,-1,1]), np.array([1,-1,1]), np.array([-1,1,1]), np.array([1,1,-1]), np.array([-1,-1,-1]), np.array([1,-1,-1]), np.array([-1,1,-1])]

[dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
print(dR)
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
        print(i / totalMeshPoints)
        i = i + 1

validMeshPoints = np.array(validMeshPoints)
invalidMeshPoints = np.array(invalidMeshPoints)

ax.scatter(validMeshPoints[:,0] / GRID_UPSCALE, validMeshPoints[:,1] / GRID_UPSCALE, validMeshPoints[:,2] / GRID_UPSCALE, s=1, c='orange')


totalMeshPoints = validMeshPoints.shape[0]
i = 0


startCoordinate = start * GRID_UPSCALE
goalCoordinate = goal * GRID_UPSCALE

graph = {}

for meshPoint in validMeshPoints.astype(np.int16):
        graph[tuple(meshPoint)] = Node(tuple(meshPoint), np.inf, np.linalg.norm(meshPoint - goalCoordinate), [])

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
     print("start is not in")
     graph[START_NODE] = Node(START_NODE, np.inf, np.linalg.norm(startCoordinate - goalCoordinate), [])
     distanceToOtherNodes = np.linalg.norm(validMeshPoints - startCoordinate, axis=1)
     indexNodesClose = np.where(distanceToOtherNodes < CLOSE_THRESHOLD)[0]
     print(indexNodesClose)
     for index in indexNodesClose:
        nearbyCoordinate = validMeshPoints[index,:]
        collisionFree = True
        for block in blocks:
            if checkCollision(startCoordinate / GRID_UPSCALE, nearbyCoordinate / GRID_UPSCALE, block):
                collisionFree = False
            if not collisionFree:
                    break
        if collisionFree:
            print(validMeshPoints[index,:])
            cost = np.linalg.norm(startCoordinate - nearbyCoordinate)
            print(cost)
            graph[START_NODE].childrenAndCosts.append((tuple(nearbyCoordinate), cost))

if GOAL_NODE not in graph:
     print("goal is not in")
     graph[GOAL_NODE] = Node(GOAL_NODE, np.inf, 0, [])
     distanceToOtherNodes = np.linalg.norm(validMeshPoints - goalCoordinate, axis=1)
     indexNodesClose = np.where(distanceToOtherNodes < CLOSE_THRESHOLD)[0]
     print(indexNodesClose)
     for index in indexNodesClose:
        nearbyCoordinate = validMeshPoints[index,:]
        collisionFree = True
        for block in blocks:
            if checkCollision(goalCoordinate / GRID_UPSCALE, nearbyCoordinate / GRID_UPSCALE, block):
                collisionFree = False
            if not collisionFree:
                    break
        if collisionFree:
            print(validMeshPoints[index,:])
            cost = np.linalg.norm(startCoordinate - nearbyCoordinate)
            print(cost)
            graph[tuple(nearbyCoordinate)].childrenAndCosts.append((GOAL_NODE, cost))

'''
nodeStartMatch = np.argmin(np.linalg.norm(validMeshPoints - (start * GRID_UPSCALE), axis=1))
START_NODE = tuple(validMeshPoints[nodeStartMatch])

nodeGoalMatch = np.argmin(np.linalg.norm(validMeshPoints - (goal * GRID_UPSCALE), axis=1))
GOAL_NODE = tuple(validMeshPoints[nodeGoalMatch])
'''

open = pqdict()
closed = []

graph[START_NODE].g = 0

open[START_NODE] = graph[START_NODE].g + graph[START_NODE].h

i = 0

while GOAL_NODE not in closed:

    print(i)

    currentNode = graph[open.pop()]

    if i == 0:
         currentNode.print()

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

while currentTraceNode != START_NODE:
    currentTraceNode = graph[currentTraceNode].parent
    shortestPath.append(currentTraceNode)

shortestPath.reverse()
shortestPath = np.array(shortestPath)

ax.plot(shortestPath[:,0] / GRID_UPSCALE,shortestPath[:,1] / GRID_UPSCALE,shortestPath[:,2] / GRID_UPSCALE,'r-')
                
plt.show(block=True)

#for node in graph.values():
#    node.print()

while False:

    directionsPossible = [np.array([dX,0,0]), np.array([0,-dY,0]), np.array([-dX,0,0]), np.array([0,dY,0]), np.array([0,0,dZ]), np.array([0,0,-dZ])]
    directionsPossible = np.round(directionsPossible, 6)

    print(directionsPossible)

    meshX = np.linspace(boundary[0,0], boundary[0,3], GRID_RESOLUTION)
    meshY = np.linspace(boundary[0,1], boundary[0,4], GRID_RESOLUTION)
    meshZ = np.linspace(boundary[0,2], boundary[0,5], GRID_RESOLUTION)

    meshX, meshY, meshZ = np.meshgrid(meshX, meshY, meshZ)

    mesh = np.vstack((meshX.flatten(),meshY.flatten(),meshZ.flatten())).T
    start = np.array([0,0,0])
    goal = np.array([4,4,0])
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    validMeshPoints = []

    for meshPoint in mesh:
        if not checkCollisionPointAABB(meshPoint, blocks) and not np.array_equal(meshPoint, start) and not np.array_equal(meshPoint, goal):
            validMeshPoints.append(meshPoint)

    #validMeshPoints.append(start)
    #validMeshPoints.append(goal)

    print(len(validMeshPoints))

    graph = {}

    for meshPoint in np.round(validMeshPoints, DECIMAL_PLACE_ROUND):
        graph[tuple(meshPoint)] = Node(meshPoint, np.inf, np.linalg.norm(meshPoint - goal), [])

    for node in graph.values():
        for directionPossible in directionsPossible:
            if tuple(np.round(node.label + directionPossible, DECIMAL_PLACE_ROUND)) in graph:
                cost = np.round(np.linalg.norm(directionPossible), DECIMAL_PLACE_ROUND)
                node.childrenAndCosts.append((tuple(np.round(node.label + directionPossible, DECIMAL_PLACE_ROUND)), cost))


    validMeshPointsArray = []


    print(graph)

    for node in graph.values():
        if len(node.childrenAndCosts) <= 2:
            validMeshPointsArray.append(node.label)
            node.print()
        

    validMeshPointsArray = np.array(validMeshPointsArray)

    ax.scatter(validMeshPointsArray[:,0], validMeshPointsArray[:,1], validMeshPointsArray[:,2], s=10, c='orange')


    plt.show(block=True)

