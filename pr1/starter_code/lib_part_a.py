from minigrid.core.world_object import Goal, Key, Door, Wall
import numpy as np

#functions for part A

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

def getObjectType(object): #return object type as a string
    if isinstance(object, type(None)):
        return "none"
    elif isinstance(object, Goal):
        return "goal"
    elif isinstance(object, Key):
        return "key"
    elif isinstance(object, Door):
        return "door" 
    elif isinstance(object, Wall):
        return "wall"

def getTypeAtCell(env, pos): #return object type at specified coordinate
    gotObject = env.grid.get(pos[0],pos[1])
    return getObjectType(gotObject)
    

def createStateVector(pos, dir, door, key): #here the custom state vector format is defined
    state = np.zeros((1,6), dtype=np.int16).flatten()
    state[0:2] = pos
    state[2:4] = dir
    state[4] = door
    state[5] = key
    return state

def getCurrentState(env, envInfo): #construct a custom state vector based on the state of the environment
    agentPosition = env.agent_pos
    agentDirection = env.dir_vec
    door = env.grid.get(envInfo["door_pos"][0], envInfo["door_pos"][1])
    doorIsOpenStatus = door.is_open
    keyPickedUpStatus = env.carrying is not None
    stateVector = createStateVector(agentPosition, agentDirection, doorIsOpenStatus, keyPickedUpStatus)
    #stateVector is a custom state defined as follows:
    # agent position:[0:1]; agent direction:[2:3]; door unlocked?:[4]; key picked up?:[5]
    return stateVector

def getNextPossibleStates(currentState, env): 


    possibleStates = np.zeros((5,6), dtype=np.int16)

    pos = currentState[0:2]
    frontDirection = currentState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]]) #the right direction unit vector, given the front is [x,y]. is [-y,x]
    leftDirection = np.array([frontDirection[1],-frontDirection[0]]) #the left direction unit vector, given the front is [x,y]. is [y,-x]
    doorOpen = currentState[4]
    pickedUpKey = currentState[5]
    
    #get the surround object types to determine the valid control inputs
    #add the direction vectors to the current position to get the cell coordinate one step in that direction
    rearObject = getTypeAtCell(env, pos + -frontDirection)
    forwardObject = getTypeAtCell(env, pos + frontDirection)
    
    if (rearObject == "none"): #if there is nothing in the front, then a valid action would be to move forward
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection #move forward motion model
        possibleStates[0,:] = possibleState
    elif (rearObject == "door" and pickedUpKey and doorOpen):
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection #move forward motion model
        possibleStates[0,:] = possibleState
    elif (rearObject == "key" and pickedUpKey):
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection #move forward motion model
        possibleStates[0,:] = possibleState

    if (forwardObject == "door" and pickedUpKey and doorOpen):
        possibleState = currentState.copy()
        possibleState[4] = 0 #motion model for an unlock door move is to change the vector state portion to unlock
        possibleStates[4,:] = possibleState
    elif (forwardObject == "key" and pickedUpKey):
        possibleState = currentState.copy() #if the key has not been picked up, it is blocking our way, so the next valid control input would be to pick it up, PK
        possibleState[5] = 0 #the next state would have the picked up key value to 1 following the motion model
        possibleStates[3,:] = possibleState

    
    turnRightState = currentState.copy()
    turnRightState[2:4] = leftDirection #motion model is to modify the agent direction
    possibleStates[2,:] = turnRightState

    turnLeftState = currentState.copy()
    turnLeftState[2:4] = rightDirection
    possibleStates[1,:] = turnLeftState

    return possibleStates #once all the control inputs have been gone through, we return the ones that are valid and will incur a stage cost of 1
    

def checkIfStateBeenVisited(state, statesVisitedList):

    #compares a target state vector to a list of state vectors that have been visited
    #returns true if that target state vector exists in the list i.e. has been visited
    diff = np.abs(np.array(statesVisitedList) - state) #subtract target state from all states in list to find vector difference
    #if difference is exactly a zero vector then there is a match
    sum = np.sum(diff, axis = 1) #a zero vector's elements will sum to 0
    return not np.all(sum) #check if there is a sum of 0, and if so that means the target state has been visited

def traceBackPolicy(controlInputs):
    #the transition chain is as follows
    # t:  ....     N-1                                N
    # chain:     [[1, TR],[2, PK],....]           [[0, MF]]
    # each time step is a list of [previousIndex, controlInput] pairs
    # previousIndex points to the index of the previous pair in the previous time

    policySequence = [] #initiate an empty optimal sequence chain to be later constructed as we traverse the chain
    timeSteps = len(controlInputs) #this represents the optimal cost-to-go from start to goal node
    previousControlIndex = 0 #for the goal step, we know the previous pair is in the 0th index
    for i in range(timeSteps-1, -1, -1): #iterate starting from end time to 0
        step = controlInputs[i][previousControlIndex] #extract the previous pair
        previousControlIndex = step[0] #we know the next previous pair index can be taken from the 1st element of the pair
        policySequence.insert(0, step[1]) #we can add to the front of the optimal sequence the control input, taken from the 2nd element of the pair
    return policySequence

def calculateOptimalPolicy(env, info, timeHorizon):
    print("CALCULATING OPTIMAL SEQUENCE")
    #core of the forward dynamic programming algorithm

    goalPos = info["goal_pos"]

    initialStates = []
    initialDirs = [np.array([0,1]), np.array([0,-1]), np.array([1,0]), np.array([-1,0])]
    initialDoors = np.array([0,1])
    initialKeys = np.array([0,1])

    for initialDir in initialDirs:
        for initialDoor in initialDoors:
            for initialKey in initialKeys:
                initialStates.append(createStateVector(goalPos, initialDir, initialDoor, initialKey))

    currentStates = np.atleast_2d(initialStates)
    visitedStates = currentStates.copy() #initialize the visited nodes to only contain the initial node
    policyDict = {}

    for t in range(timeHorizon): #we will iterate up to a time horizon, and assume that the optimal cost-to-arrive at the current nodes is equal to t
        
        nextStates = []
        for currentStateIndex in range(currentStates.shape[0]): #iterate through all the "surviving" nodes with a cost-to-arrive of t; these still have potential to be part of the shortest path
            currentState = currentStates[currentStateIndex,:]
            nextPossibleStates = getNextPossibleStates(currentState, env) #for each surviving node, find all the potential next nodes with corresponding control inputs
            for controlInput in range(5): #iterate through the 5 possible control inputs
                nextPossibleState = nextPossibleStates[controlInput,:] #get the next node corresponding to that control input
                if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))): #check if the stage cost for that control input is not infinite
                    if not checkIfStateBeenVisited(nextPossibleState, visitedStates): #check if the state has been visited because then the cost-to-arrive is not optimal
                        #at this point, the next node has been checked to satisfy both conditions such that the stage cost is 1, or else it would be infinite and not considered
                        visitedStates = np.vstack((visitedStates, nextPossibleState)) #add this next node to the visited states
                        nextStates.append(nextPossibleState) #add this next node to the next states list, which will be the future surviving current states
                        policyDict[np.array2string(nextPossibleState)[1:-1]] = controlInput

        currentStates = np.array(nextStates)

        if (len(nextStates) == 0):
            break

    return policyDict