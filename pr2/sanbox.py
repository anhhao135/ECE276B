import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from main import *
from lib import *
from pqdict import pqdict


boundary, blocks = load_map('./maps/single_cube.txt')
boundary = boundary.flatten()

print(boundary)

fig = plt.figure()
ax = plt.axes(projection='3d')

graph = []

for i in range(50):
    #point = sampleCSpace(boundary)
    point = sampleCFreeSpace(boundary, blocks)
    ax.scatter3D(point[0], point[1], point[2], c='yellow')
    graph.append(point)


graph = np.array(graph)


testPoint = np.array([2.5,2.5,5])
ax.scatter3D(testPoint[0], testPoint[1], testPoint[2], c='red')

#closePointsIndex = nearNodes(testPoint, graph, 7)
closePointsIndex = nearestNode(testPoint, graph)
point = graph[closePointsIndex]
ax.scatter3D(point[0], point[1], point[2], c='green')
plt.show(block=True)