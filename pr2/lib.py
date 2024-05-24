import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from main import *
from pqdict import pqdict

#Hao Le A15547504 ECE 276B SP24 Project 2
#library for neccessary functions to perform search and sampling-based motion planning

#core Node class for weighted A* search
class Node():
    def __init__(self, label, g, h, childrenAndCosts):
        self.label = label #this will be the high level "name" of the node used to look it up in the graph; this will be the node coordinates in 3-D space converted to a hashable tuple
        self.g = g #for label correction
        self.h = h #for heuristic value
        self.parent = None #for A* backtrack
        self.childrenAndCosts = childrenAndCosts #a list of tuples that describe the edges outgoing from a node: (child label, cost)
    def print(self): #print out members of a node for debugging
        print("--------------------------------------")
        print("label: " + str(self.label))
        print("g: " + str(self.g))
        print("h: " + str(self.h))
        print("parent: " + str(self.parent))
        print("(children,cost): " + str(self.childrenAndCosts))
        print("--------------------------------------")

def checkCollisionPointAABB(point, blocks):
    #function for checking if a single point in 3-D space collides with an axis-aligned bounding box
    collision = False
    for block in blocks: #we check if all of the point's components are fully contained in the min-max ranges of the AABB
        if point[0] > block[0] and point[0] < block[3] and\
            point[1] > block[1] and point[1] < block[4] and\
            point[2] > block[2] and point[2] < block[5]:
                collision = True
                break
    return collision
    
def checkCollision(point1, point2, block):
    #function for checking if a line segment, described by two 3-D points, collides with an AABB
    #this is essentially an extended Slab method that can work with finite-length segments instead of infinite length rays
    #more detailed description can be found in the project report
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


    if (0 > t_close and 0 > t_far) or (1 < t_close and 1 < t_far): #this is the modification for finite length segments; if both intersection parameters fall out of [0,1], this indicates no part of the segment collides with the AABB
        return False
    else:
        return t_close <= t_far


def checkCollisionFreeEntireMap(point1, point2, blocks):
    #check for finite-length line segment collision with any of the obstacles in the map
    collisionFree = True
    for block in blocks:
        if checkCollision(point1, point2, block):
            collisionFree = False
        if not collisionFree:
            break
    return collisionFree

def weightedAStarHeuristic(point, goal, epsilon):
    #weighted A* heuristic is just the Euclidian distance of the node to the goal node multiplied by an epsilon value
    return epsilon * np.linalg.norm(point - goal)

def weightedAStar(graph, START_NODE, GOAL_NODE, horizon):
    #core weighted A* algorithm
    open = pqdict() #open list is implemented as a priority queue
    closed = [] #closed list is just a normal list
    open[START_NODE] = graph[START_NODE].g + graph[START_NODE].h #add the start node to the open list where its f value is g + h

    i = 0

    while GOAL_NODE not in closed: #loop until goal node is found in the closed list (expanded)

        if i == horizon or len(open) == 0: #terminate algorithm if goal was not reached in time or if open list is empty which indicates the path is incomplete
             print("failed")
             return None, 0

        currentNode = graph[open.pop()] #the current node is the node with the lowest f value in the open list i.e. the first one out of the priority queue
        closed.append(currentNode.label) #insert the expanded node from the open list into the closed list

        for (child, cost) in currentNode.childrenAndCosts:
            if child not in closed: #iterate through all unexpanded children of the current node
                if graph[child].g > currentNode.g + cost:
                    graph[child].g = currentNode.g + cost #label correcting algorithm updates the g value if a lower cost path has been found to the node
                    graph[child].parent = currentNode.label
                    open[child] = graph[child].g + graph[child].h #add the child to the open list with the f value, if the child is already in the open list, its priority will be automatically updated
        
        i = i + 1

    #once the algorithm terminates, goal node has been found
    #now we trace the shortest path back to the start via the parents

    shortestPath = [GOAL_NODE]
    currentTraceNode = GOAL_NODE
    totalCost = 0

    while currentTraceNode != START_NODE: #stop tracing once the start node has been reached
        cost = np.linalg.norm(np.array(graph[currentTraceNode].label) - np.array(graph[currentTraceNode].parent)) #the cost is the euclidian distance
        totalCost = totalCost + cost #accumulate the cost
        currentTraceNode = graph[currentTraceNode].parent #traverse to the next parent
        shortestPath.append(currentTraceNode) #record down the shortest path hops

    shortestPath.reverse() #reverse the shortest path tracker since it started from the goal
    shortestPath = np.array(shortestPath)

    return shortestPath, totalCost #return shortest path node hops and the total cost


def searchBasedPlan(start, goal, mapDirectory, GRID_UPSCALE, horizon, epsilon):
    #search based motion planner that constructs the graph in a uniform, discrete, deterministic manner
    #then runs weighted A* on the graph to find the shortest path from start to goal configuration

    startTime = time.time() #for benchmarking performance

    CLOSE_THRESHOLD = 2 #radius to connect start and goal nodes to the constructed graph later for completeness
    boundary, blocks = load_map(mapDirectory) #load map

    #discretize the continuous space by the resolution specified in GRID_UPSCALE
    meshX = np.arange(boundary[0,0] * GRID_UPSCALE, (boundary[0,3] * GRID_UPSCALE) + 1)
    meshY = np.arange(boundary[0,1] * GRID_UPSCALE, (boundary[0,4] * GRID_UPSCALE) + 1)
    meshZ = np.arange(boundary[0,2] * GRID_UPSCALE, (boundary[0,5] * GRID_UPSCALE) + 1)

    meshX, meshY, meshZ = np.meshgrid(meshX, meshY, meshZ)

    mesh = np.vstack((meshX.flatten(),meshY.flatten(),meshZ.flatten())).T

    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    #now we create the 26 possible directions originating from a point
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1) #(0,0,0) is not valid since it is just staying still so delete it
    directionsPossible = dR.T

    validMeshPoints = []
    invalidMeshPoints = []

    totalMeshPoints = mesh.shape[0]
    i = 0

    for meshPoint in mesh:
            if not checkCollisionPointAABB(meshPoint / GRID_UPSCALE, blocks) and checkCollisionPointAABB(meshPoint / GRID_UPSCALE, boundary): #if the discrete point does not collide with an AABB, it is part of Cfree
                validMeshPoints.append(meshPoint)
            else: #else it is part of Cobstacle so don't bother including it
                invalidMeshPoints.append(meshPoint)
            i = i + 1

    validMeshPoints = np.array(validMeshPoints)
    invalidMeshPoints = np.array(invalidMeshPoints)

    ax.scatter(validMeshPoints[:,0] / GRID_UPSCALE, validMeshPoints[:,1] / GRID_UPSCALE, validMeshPoints[:,2] / GRID_UPSCALE, s=5, c='orange') #plot the discretized grid of Cfree


    totalMeshPoints = validMeshPoints.shape[0]
    i = 0

    startCoordinate = start * GRID_UPSCALE #we also need to scale the coordinates up by the grid resolution to make sense in the graph space
    goalCoordinate = goal * GRID_UPSCALE

    graph = {} #initalized empty graph as dictionary where key is the node coordinate and value is the node object

    for meshPoint in validMeshPoints.astype(np.int16):
            graph[tuple(meshPoint)] = Node(tuple(meshPoint), np.inf, weightedAStarHeuristic(meshPoint, goalCoordinate, epsilon), []) #instantiate all grid points as a node object where the label is the lookup-able tuple of its coordinate; we also set the g as infinited, and add its heuristic.

    for node in graph.values():
            print(i / totalMeshPoints) #print the progress of adding the edges for each node
            i = i + 1
            for directionPossible in directionsPossible: #go through all the possible edges for a node corresponding to the 26 possible directions
                if tuple(node.label + directionPossible.astype(np.int16)) in graph: #check if when we move in that direction, it leads to an existing graph node
                    if checkCollisionFreeEntireMap(np.array(node.label) / GRID_UPSCALE, (np.array(node.label) + directionPossible.astype(np.int16)) / GRID_UPSCALE, blocks): #check if the line segment between current node and adjacent node is collision free; if so, then add the edge
                        cost = np.linalg.norm(directionPossible) #edge cost is simply the euclidian distance
                        node.childrenAndCosts.append((tuple(node.label + directionPossible.astype(np.int16)), cost)) #add the edge to the node object as a tuple (child, cost)


    START_NODE = tuple(startCoordinate)
    GOAL_NODE = tuple(goalCoordinate)

    #now we include both start and goal nodes in the discrete graph if it was "missed" initially by the discretization process. This is useful for non-integer coordinates since it creates a complete graph from start to finish without approximating start and stop conditions via within-radius check for termination

    if START_NODE not in graph: #check if the start node was missed in the discrete graph, if so let's add it and add it's edges
        graph[START_NODE] = Node(START_NODE, np.inf, weightedAStarHeuristic(start, goalCoordinate, epsilon), [])
        indexNodesClose = nearNodes(startCoordinate, validMeshPoints, CLOSE_THRESHOLD) #find all the nodes that are close to it within a range that can be linked to via an edge
        for index in indexNodesClose: #now we check which close nodes can actually be linked to collision-free
            nearbyCoordinate = validMeshPoints[index,:]
            collisionFree = True
            for block in blocks:
                if checkCollision(startCoordinate / GRID_UPSCALE, nearbyCoordinate / GRID_UPSCALE, block):
                    collisionFree = False
                if not collisionFree:
                        break
            if collisionFree: #add the edge if collision-free
                cost = np.linalg.norm(startCoordinate - nearbyCoordinate)
                graph[START_NODE].childrenAndCosts.append((tuple(nearbyCoordinate), cost))

    #same process for the goal node
    if GOAL_NODE not in graph:
        graph[GOAL_NODE] = Node(GOAL_NODE, np.inf, 0, [])
        indexNodesClose = nearNodes(goalCoordinate, validMeshPoints, CLOSE_THRESHOLD)
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

    graph[START_NODE].g = 0 #start node's g is 0

    graphConstructionFinishTime = time.time()

    shortestPath, totalCost = weightedAStar(graph, START_NODE, GOAL_NODE, horizon) #run weighted A* on the constructed graph

    graphSearchFinishTime = time.time()

    ax.plot(shortestPath[:,0] / GRID_UPSCALE,shortestPath[:,1] / GRID_UPSCALE,shortestPath[:,2] / GRID_UPSCALE,'r-') #plot the shortest path found

    #return some info on the parameters and time results for plotting
    returnString = "Searched-based plan, weighted A*\ngrid upscale: {GRID_UPSCALE}, epsilon: {epsilon}, graph construction time: {constructionTime:.5f}, graph search time: {searchTime:.10f}, shortest path length: {pathLength:.5f}, graph size: {graphSize}".format(GRID_UPSCALE = GRID_UPSCALE, epsilon = epsilon, constructionTime = graphConstructionFinishTime - startTime, searchTime = graphSearchFinishTime - graphConstructionFinishTime, pathLength = totalCost / GRID_UPSCALE, graphSize = len(graph))

    return totalCost / GRID_UPSCALE, True, returnString



#primitive functions for sampling-based graph construction

def sampleCSpace(boundary): #get a random point in the C space
     minX = boundary[0]
     minY = boundary[1]
     minZ = boundary[2]
     maxX = boundary[3]
     maxY = boundary[4]
     maxZ = boundary[5]

    #generate random x y z values within the bounds of the space
     sampleX = np.random.uniform(minX, maxX)
     sampleY = np.random.uniform(minY, maxY)
     sampleZ = np.random.uniform(minZ, maxZ)

     return np.array([sampleX, sampleY, sampleZ])

def sampleCFreeSpace(boundary, blocks): #get a random point in C free
    sampledPoint = sampleCSpace(boundary) #get a random point which may or may not collide with an AABB
    while checkCollisionPointAABB(sampledPoint, blocks): #if it does, get another random point until it is collision free
        sampledPoint = sampleCSpace(boundary)
    return sampledPoint

def nearestNode(point, graph): #return the nearest graph node in terms of euclidian distance
    graph = np.array(graph)
    distanceToOtherNodes = np.linalg.norm(graph - point, axis=1)
    indexNodeClosest = np.argmin(distanceToOtherNodes)
    return indexNodeClosest

def nearNodes(point, graph, radius): #return a set of graph nodes within a radius of the current node
    graph = np.array(graph)
    distanceToOtherNodes = np.linalg.norm(graph - point, axis=1) #get list of distances to all other nodes in the graph
    indexesNodesClose = np.where(distanceToOtherNodes < radius)[0] #filter out nodes that are of distance larger than the radius
    return indexesNodesClose

def steerToNode(point1, point2, epsilon): #return point that is closest to the target point without leaving the radius (epsilon) to the current point
    distance = np.linalg.norm(point2 - point1) #find the distance vector
    if distance <= epsilon: #if the target point is within the radius of the current point
        return point2 #then we can simply return the target point
    else: #if its further out
        normalizedVector = (point2 - point1) / distance #the closest point is along the distance vector
        return point1 + (epsilon * normalizedVector) #such that the distance to the current point is the radius
    
def visualizeGraph(graph, ax): #for visually plotting out the graph tree structure
    for node in graph.values(): #iterate through all the nodes in the graph and draw out its edges
        nodePoint = np.array(node.label)
        for (child, cost) in node.childrenAndCosts:
            drawVector = np.array([nodePoint, np.array(child)])
            ax.plot(drawVector[:,0],drawVector[:,1],drawVector[:,2], color='orange', alpha = 0.6)


def samplingBasedPlanBiDirectionalRRT(start, goal, mapDirectory, nodesToSample, horizon, epsilon, steerEpsilon):
    #sampling based motion planner that constructs the graph in a stochastic manner
    #then runs weighted A* on the graph to find the shortest path from start to goal configuration
    startTime = time.time()

    #below is the implementation of the RRT-Connect algorithm for stochastic graph construction
    #more details are discussed in the project report

    def extend(graphList, graphDict, point, steerEpsilon, blocks):
        #subroutine to attempt to connect a random point to the nearest point in the existing graph and reports if 1. a collision occurs 2. a collision-free advanced towards the point has been made or 3. point has been successfully reached
        pointNearest = graphList[nearestNode(point, graphList)]
        pointNew = steerToNode(pointNearest, point, steerEpsilon)
        if checkCollisionFreeEntireMap(pointNearest, pointNew, blocks):
            if tuple(pointNew) not in graphDict:
                graphDict[tuple(pointNew)] = Node(tuple(pointNew), np.inf, weightedAStarHeuristic(pointNew, goal, epsilon), []) #if the new point is not in the graph, add it
            graphList.append(pointNew)
            cost = np.linalg.norm(pointNearest - pointNew)
            #add the new edges
            graphDict[tuple(pointNearest)].childrenAndCosts.append((tuple(pointNew), cost))
            graphDict[tuple(pointNew)].childrenAndCosts.append((tuple(pointNearest), cost))
            if np.array_equal(pointNew, point): #if the new point is equal, that means it was reachable via the steering
                return "reached", pointNew
            else:
                return "advanced", pointNew
        return "trapped", pointNew #if there was a collision, then the two points cannot connect any further

    def connect(graphList, graphDict, point, steerEpsilon, blocks):
        #attempts to connect a random point to the nearest point on the graph until a collision occurs
        status, pointNew = extend(graphList, graphDict, point, steerEpsilon, blocks)
        while status == "advanced":
            status, pointNew = extend(graphList, graphDict, point, steerEpsilon, blocks)
        return status

    boundary, blocks = load_map(mapDirectory)
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
    boundary = boundary.flatten()

    START_NODE = tuple(start)
    GOAL_NODE = tuple(goal)

    #for bi-directional RRT, we keep track of the trees that originate either from the start or goal node
    treeAList = []
    treeBList = []
    graphDict = {}

    treeAList.append(start)
    treeBList.append(goal)
    graphDict[START_NODE] = Node(START_NODE, 0, weightedAStarHeuristic(start, goal, epsilon), [])
    graphDict[GOAL_NODE] = Node(GOAL_NODE, np.inf, 0, [])
    shortestPath = None
    totalCost = None

    graphConstructionFinishTime = time.time()
    graphSearchFinishTime = time.time()

    #core RRT-connect algorithm

    for i in range(nodesToSample): #sample a finite number of times and try to make a complete graph from that
        pointRand = sampleCFreeSpace(boundary, blocks) #get random Cfree point
        status, pointNew = extend(treeAList, graphDict, pointRand, steerEpsilon, blocks) #attempt to grow tree A to that point
        if status != "trapped": #if there is a collision-free connection or advancement
            if connect(treeBList, graphDict, pointNew, steerEpsilon, blocks) == "reached": #now we attempt to connect the two trees for every iteration; this is the magic of RRT-connect as it can connect through narrow passage ways onces both trees are visible to each other
                #if a connection is made, then the graph is complete from start to goal for sure
                graphConstructionFinishTime = time.time()
                #run weighted A* on the constructed complete graph
                shortestPath, totalCost = weightedAStar(graphDict, START_NODE, GOAL_NODE, horizon)
                graphSearchFinishTime = time.time()
                break
        treeAList, treeBList = treeBList, treeAList #swap the trees so we can alternative growth between trees for each iteration for equal growth attempts

    graphSearchFinishTime = time.time()
    ax.view_init(azim=-90, elev=90)
    visualizeGraph(graphDict, ax)

    ax.plot(shortestPath[:,0],shortestPath[:,1],shortestPath[:,2],'r-')

    returnString = "Sampling-based plan, bi-directional RRT-Connect, weighted A*\nepsilon: {epsilon}, RRT steer epsilon: {steerEpsilon}, graph construction time: {constructionTime:.5f}, graph search time: {searchTime}, shortest path length: {pathLength:.5f}, graph size: {graphSize}".format(epsilon = epsilon, steerEpsilon = steerEpsilon, constructionTime = graphConstructionFinishTime - startTime, searchTime = graphSearchFinishTime - graphConstructionFinishTime, pathLength = totalCost, graphSize = len(graphDict))

    return totalCost, True, returnString

    

