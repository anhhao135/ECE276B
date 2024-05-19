import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from main import *
from lib import *
from pqdict import pqdict

GRID_UPSCALE = 1 #meters
DECIMAL_PLACE_ROUND = 3

boundary, blocks = load_map('./maps/my_cube.txt')
print(boundary)

meshX = np.arange(boundary[0,0] * GRID_UPSCALE, (boundary[0,3] * GRID_UPSCALE) + 1)
meshY = np.arange(boundary[0,1] * GRID_UPSCALE, (boundary[0,4] * GRID_UPSCALE) + 1)
meshZ = np.arange(boundary[0,2] * GRID_UPSCALE, (boundary[0,5] * GRID_UPSCALE) + 1)

meshX, meshY, meshZ = np.meshgrid(meshX, meshY, meshZ)

mesh = np.vstack((meshX.flatten(),meshY.flatten(),meshZ.flatten())).T
start = np.array([0.3,0.3,0.3])
goal = np.array([3.3,3.3,0.3])
fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

print(blocks)

directionsPossible = [np.array([1,0,0]), np.array([0,-1,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([0,0,-1]), np.array([1,1,0]), np.array([-1,-1,0]), np.array([1,-1,0]), np.array([-1,1,-0]), np.array([1,1,1]), np.array([-1,-1,1]), np.array([1,-1,1]), np.array([-1,1,1]), np.array([1,1,-1]), np.array([-1,-1,-1]), np.array([1,-1,-1]), np.array([-1,1,-1])]

validMeshPoints = []
invalidMeshPoints = []

for meshPoint in mesh:
        if not checkCollisionPointAABB(meshPoint / GRID_UPSCALE, blocks):
            validMeshPoints.append(meshPoint)
        else:
            invalidMeshPoints.append(meshPoint)

validMeshPoints = np.array(validMeshPoints)
invalidMeshPoints = np.array(invalidMeshPoints)

ax.scatter(validMeshPoints[:,0] / GRID_UPSCALE, validMeshPoints[:,1] / GRID_UPSCALE, validMeshPoints[:,2] / GRID_UPSCALE, s=10, c='orange')


graph = {}

for meshPoint in validMeshPoints.astype(np.int16):
        graph[tuple(meshPoint)] = Node(tuple(meshPoint), np.inf, np.linalg.norm(meshPoint - goal), [])

for node in graph.values():
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
                    print(cost)
                    node.childrenAndCosts.append((tuple(node.label + directionPossible.astype(np.int16)), cost))


for node in graph.values():
    node.print()



GOAL_NODE = tuple(np.array([3,3,0], dtype = np.int16) * GRID_UPSCALE)
START_NODE = tuple(np.array([0,0,0], dtype = np.int16) * GRID_UPSCALE)

open = pqdict()
closed = []

graph[START_NODE].g = 0

open[START_NODE] = graph[START_NODE].g + graph[START_NODE].h

i = 0

while GOAL_NODE not in closed:

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

