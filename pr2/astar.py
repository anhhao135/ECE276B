
# priority queue for OPEN list
from pqdict import pqdict
import math
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


GOAL_NODE = 5
START_NODE = 0

graph = {}
graph[0] = Node(0, 0, 3, [(2,1)])
graph[2] = Node(2, np.inf, 2, [(1,2),(4,1)])
graph[1] = Node(1, np.inf, 1, [(5,2)])
graph[3] = Node(3, np.inf, 1, [(5,1)])
graph[4] = Node(4, np.inf, 2, [(3,3)])
graph[5] = Node(5, np.inf, 0, [])

open = pqdict()
closed = []

open[graph[0].label] = graph[0].g + graph[0].h

for node in graph.values():
    node.print()

i = 0

while GOAL_NODE not in closed:

    print(i)

    currentNode = graph[open.pop()]

    print("current node: " + str(currentNode.label))

    closed.append(currentNode.label)
    for (child, cost) in currentNode.childrenAndCosts:
        if child not in closed:
            if graph[child].g > currentNode.g + cost:
                graph[child].g = currentNode.g + cost
                graph[child].parent = currentNode.label
                open[child] = graph[child].g + graph[child].h

    for node in graph.values():
        node.print()
    
    print("open: " + str(open))
    print("closed: " + str(closed))

    i = i + 1

shortestPath = [GOAL_NODE]

currentTraceNode = GOAL_NODE

while currentTraceNode != START_NODE:
    currentTraceNode = graph[currentTraceNode].parent
    shortestPath.append(currentTraceNode)

shortestPath.reverse()

print(shortestPath)
                




