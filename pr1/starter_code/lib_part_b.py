import numpy as np
from tqdm import tqdm
 


NONE = 0
DOOR = 1
KEY = 2
GOAL = 3
WALL = 4


def constructRandomMap(goalLocations, keyLocations, doorLocations, dimension): #we create a probabilistic representation of the map so we can account for different scenarios of key/goal/doors and act accordingly with control inputs

    randomMap = np.zeros((dimension,dimension), dtype=np.int16)
    wallColumn = doorLocations[0][0]

    randomMap[wallColumn,:] = WALL #we know where the wall column will be

    for goalLocation in goalLocations:
        randomMap[goalLocation[0], goalLocation[1]] = GOAL #mark where goals can potentially be

    for keyLocation in keyLocations:
        randomMap[keyLocation[0], keyLocation[1]] = KEY #mark where keys can potentially be

    for doorLocation in doorLocations:
        randomMap[doorLocation[0], doorLocation[1]] = DOOR #mark where we know the doors to be located, but we do not know the lock states of

    surroundingWallsMap = np.full((dimension+2,dimension+2), WALL, dtype=np.int16) #pad around the map wall so the stage cost of going "off" the real map is infinite and should not happen
    surroundingWallsMap[1:dimension+1, 1:dimension+1] = randomMap

    return surroundingWallsMap

def createStateVector(currentPos, currentDir, goalPos, keyPos, doorState, currentKeyPickedUp, door1UnlockAttempted, door2UnlockAttempted): #here we create a custom current state vector based on information from the environment at a certain time step
    state = np.zeros((1,13), dtype=np.int16).flatten()
    state[0:2] = currentPos
    state[2:4] = currentDir
    state[4:6] = goalPos
    state[6:8] = keyPos
    state[8:10] = doorState
    state[10] = currentKeyPickedUp
    state[11] = door1UnlockAttempted
    state[12] = door2UnlockAttempted
    return state #this state can then be mapped to a control input from the optimal policy



def createEndGoalStates(goalState): #create a goal state with the agent facing all four different directions
    #this is to discourage other exploratory branches from reaching the goal as their path will be sub-optimal
    state1 = goalState.copy()
    state1[2:4] = np.array([0,-1],dtype=np.int16)
    
    state2 = goalState.copy()
    state2[2:4] = np.array([1,0],dtype=np.int16)
    state3 = goalState.copy()
    state3[2:4] = np.array([0,1],dtype=np.int16)
    state4 = goalState.copy()
    state4[2:4] = np.array([-1,0],dtype=np.int16)

    visitedGoalStates = np.zeros((4,13), dtype=np.int16)
    visitedGoalStates[0,:] = state1
    visitedGoalStates[1,:] = state2
    visitedGoalStates[2,:] = state3
    visitedGoalStates[3,:] = state4
    return visitedGoalStates


def getNextPossibleStates(currentState, randomMap, DOOR_1_POS): #given a state vector node, what are the possible next states?
    #the state cost to a possible state is simply 1, the cost of a valid control input
    #otherwise, the stage cost is infinite and the control input is not considered to be optimal for the value function at time t
    #for a given current state, go through all the 5 possible control inputs and determine which is valid and the motion model output to get the next state vector
    #the difference here from part A is now we use a random map to query the potential of items exisitng at a cell instead of it always being there

    #extract information about the environment and the agent from the custom current state vector
    pos = currentState[0:2]
    frontDirection = currentState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]])
    leftDirection = np.array([frontDirection[1],-frontDirection[0]])
    frontPos = pos + frontDirection
    rearPos = pos - frontDirection

    goalPos = currentState[4:6]
    keyPos = currentState[6:8]
    doorState = currentState[8:10]
    keyPickedUp = currentState[10]
    door1UnlockAttempt = currentState[11]
    door2UnlockAttempt = currentState[12]
    frontObject = NONE
    rearObject = NONE
    possibleStates = np.zeros((6,13), dtype=np.int16)

    #we now deduce what the front object could potentially be based on the current state

    if (np.array_equal(frontPos,keyPos)): #if there is a potential for the cell in front of us to be a key and we have not picked up a key
        frontObject = KEY #consider it a key if it has been randomly placed there
    elif (randomMap[frontPos[0], frontPos[1]] == DOOR): #if the agent is facing a door
        frontObject = DOOR #else, consider the door to be closed
    elif (randomMap[frontPos[0], frontPos[1]] == WALL): #if the front object is a wall, then it is for sure a wall
        frontObject = WALL

    if (randomMap[rearPos[0], rearPos[1]] == DOOR):
        rearObject = DOOR
    if (randomMap[rearPos[0], rearPos[1]] == WALL):
        rearObject = WALL
    elif (np.array_equal(rearPos,keyPos)):
        rearObject = KEY

    #now the expected object in front has been deduced and we can perform the motion model on the current state for all 5 control inputs

    if (rearObject == NONE):
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection
        possibleStates[0,:] = possibleState
    elif (rearObject == DOOR):
        if (np.array_equal(rearPos,DOOR_1_POS)):
            if (doorState[0]):
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] - frontDirection
                possibleStates[0,:] = possibleState
            elif (keyPickedUp and door1UnlockAttempt):
                
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] - frontDirection
                possibleStates[0,:] = possibleState
        else:
            if (doorState[1]):
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] - frontDirection
                possibleStates[0,:] = possibleState
            elif (keyPickedUp and door2UnlockAttempt):
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] - frontDirection
                possibleStates[0,:] = possibleState
    elif (rearObject == KEY and keyPickedUp):
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection
        possibleStates[0,:] = possibleState


    if (frontObject == DOOR):
        if (np.array_equal(frontPos,DOOR_1_POS)):
            if (not doorState[0] and door1UnlockAttempt):
                possibleState = currentState.copy()
                possibleState[11] = 0 #attempt at unlocking door number 1
                possibleStates[4,:] = possibleState
        else:
            if (not doorState[1] and door2UnlockAttempt):
                possibleState = currentState.copy()
                possibleState[12] = 0 #attempt at unlocking door number 2
                possibleStates[4,:] = possibleState
    elif (frontObject == KEY and keyPickedUp):
        possibleState = currentState.copy()
        possibleState[10] = 0 #pick it up
        possibleStates[3,:] = possibleState


    #for any state, it is always valid to turn left or right in place at a cost of 1
    turnRightState = currentState.copy()
    turnRightState[2:4] = leftDirection
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

def calculateSingleOptimalPolicy(timeHorizon, goalLocations, keyLocations, doorLocations, dimension):

    randomMap = constructRandomMap(goalLocations, keyLocations, doorLocations, dimension) #create the probabilistic map based on the random generation parameters

    print(randomMap.T)

    initialGoalLocations = np.array(goalLocations, dtype=np.int16) + np.array([1,1])
    initialKeyLocations = np.array(keyLocations, dtype=np.int16) + np.array([1,1])
    initialDirs = [np.array([0,1]), np.array([0,-1]), np.array([1,0]), np.array([-1,0])]
    initialDoorStates = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
    #initialDoorUnlockAttemptedStates = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
    initialDoorUnlockAttemptedStates = [np.array([0,0])]
    initialKeyPickedUpStates = np.array([0,1])

    initialStates = []

    for initialGoalLocation in initialGoalLocations:
        for initialKeyLocation in initialKeyLocations:
            for initialDir in initialDirs:
                for initialDoorState in initialDoorStates:
                    for initialDoorUnlockAttemptedState in initialDoorUnlockAttemptedStates:
                        for initialKeyPickedUpState in initialKeyPickedUpStates:
                            initialStates.append(createStateVector(initialGoalLocation, initialDir, initialGoalLocation, initialKeyLocation, initialDoorState, initialKeyPickedUpState, initialDoorUnlockAttemptedState[0], initialDoorUnlockAttemptedState[1]))

    
    #testState = createStateVector(np.array([3,1]), np.array([0,-1]), np.array([3,1]), np.array([1,1]), np.array([0,0]), 1, 1, 0)
    #initialStates = [testState]

    currentStates = np.atleast_2d(initialStates)
    visitedStates = currentStates.copy() #initialize the visited nodes to only contain the initial node
    policyDict = {}

    DOOR_1_POS = doorLocations[0] + np.array([1,1])
    #DOOR_2_POS = doorLocations[1] + np.array([1,1])

    for t in tqdm (range (timeHorizon), desc="time step"): #we will iterate up to a time horizon, and assume that the optimal cost-to-arrive at the current nodes is equal to t
        print("------------------------------------------")
        print("time: " + str(t))
        print("current state count: " + str(currentStates.shape[0]))
        
        nextStates = []
        for currentStateIndex in tqdm (range (currentStates.shape[0]), desc="current state"): #iterate through all the "surviving" nodes with a cost-to-arrive of t; these still have potential to be part of the shortest path
            currentState = currentStates[currentStateIndex,:]
            nextPossibleStates = getNextPossibleStates(currentState, randomMap, DOOR_1_POS) #for each surviving node, find all the potential next nodes with corresponding control inputs
            for controlInput in range(5): #iterate through the 5 possible control inputs
                nextPossibleState = nextPossibleStates[controlInput,:] #get the next node corresponding to that control input
                if not (np.array_equal(nextPossibleState, np.zeros(13, dtype=np.int16))): #check if the stage cost for that control input is not infinite
                    if not checkIfStateBeenVisited(nextPossibleState, visitedStates): #check if the state has been visited because then the cost-to-arrive is not optimal
                        #at this point, the next node has been checked to satisfy both conditions such that the stage cost is 1, or else it would be infinite and not considered
                        visitedStates = np.vstack((visitedStates, nextPossibleState)) #add this next node to the visited states
                        nextStates.append(nextPossibleState) #add this next node to the next states list, which will be the future surviving current states
                        offsetNextPossibleState = nextPossibleState - np.array([1,1,0,0,1,1,1,1,0,0,0,0,0]) #correct offset
                        policyDict[np.array2string(offsetNextPossibleState)[1:-1]] = controlInput

        currentStates = np.array(nextStates)
        print("-----------------")
        print("current states")
        print(currentStates)
        print("-----------------")
        print("-----------------")
        print("visited states")
        print(visitedStates)
        print("-----------------")
        print("-----------------")
        print("policy dict")
        print(policyDict)
        print("-----------------")
        print("------------------------------------------")
        if (len(nextStates) == 0):
            break
    return policyDict

    while False:

        #add [1,1] to all absolute coordinates because our random map is padded with surrounding wall; this is so the motion model can detect boundaries outside of the real map
        initialPos = initialPos + np.array([1,1])
        DOOR_1_POS = doorLocations[0] + np.array([1,1])
        DOOR_2_POS = doorLocations[1] + np.array([1,1])
        goalLocations = np.array(goalLocations, dtype=np.int16) + np.array([1,1])
        keyLocations = np.array(keyLocations, dtype=np.int16) + np.array([1,1])

        doorStates = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.int16) #the 4 possible door lock states given we have 2 doors

        initialStates =  [] #initialize the initial states list

        #here we construct all possible initial states based on the random map generation parameters
        for goalLocation in goalLocations:
            for keyLocation in keyLocations:
                for doorState in doorStates:
                    initialState = createStateVector(initialPos, initialDir, goalLocation, keyLocation, doorState, 0)
                    initialStates.append(initialState)
        #the custom state vector is formatted as follows, it is an extension of the format in part A
        # agent position:[0:1]; agent direction:[2:3]; goal position:[4:5]; key position:[6:7]; initial door lock state [8:9]; key picked up? [10]; door 1 unlock attempted? [11]; door 2 unlock attempted? [12]

        currentStates = np.atleast_2d(np.array(initialStates)) #initialize the current states as the possible initial states
        visitedStates = currentStates.copy() #add the current states to the visited states tracker

        policyDict = {} #initialize an empty policy dictionary where
        #key = encoding of node-to-node(state-to-state) transition at a time step
        #value = control input to induce the transition

        #we will execute a variant of the forward dynamic programming algorithm used in part A, except there are multiple initial states, and the map contains all possible locations of keys and goals
        #this means each unique state and time pair will be mapped to different control inputs such that the optimal action sequence will be produced for any given random map, assuming the map does not change midway through

        for t in range(timeHorizon):

            if len(currentStates) == 0:#terminate the algorithm once there are no surviving branches and the current states are empty
                break

            nextStates = []
            initialStatesWithGoalFound = [np.array([-1,-1,-1,-1,-1,-1])] #keep track of which initial state has reached the goal, initialize with non-existent state

            for currentStateIndex in range(currentStates.shape[0]): #iterate through all the surviving nodes up to time step t
                currentState = currentStates[currentStateIndex,:]
                nextPossibleStates = getNextPossibleStates(currentState, randomMap, DOOR_1_POS) #for each surviving node, find all the potential next nodes with corresponding control inputs
                for controlInput in range(5): #iterate through the 5 possible control inputs
                    nextPossibleState = nextPossibleStates[controlInput,:] #get the next node corresponding to that control input
                    if not (np.array_equal(nextPossibleState, np.zeros(13, dtype=np.int16))): #check if the stage cost for that control input is not infinite
                        if not checkIfStateBeenVisited(nextPossibleState, visitedStates):#check if the state has been visited because then the cost-to-arrive is not optimal
                            goalReached = np.array_equal(nextPossibleState[0:2], nextPossibleState[4:6]) #check if the next possible state is a goal state by examining if the current position is equal to the goal position in the state vector
                            if (goalReached):
                                if not checkIfStateBeenVisited(nextPossibleState[4:10], initialStatesWithGoalFound): #check if the initial map condition has a goal located previously; if not, keep track of it so other surviving branches with the same initial state will get killed off
                                    initialStatesWithGoalFound.append(nextPossibleState[4:10]) 
                                endGoalStates = createEndGoalStates(nextPossibleState) #inject into the visited states list all four possible end goal states for the initial state so that other branches being explored are killed off
                                visitedStates = np.vstack((visitedStates, endGoalStates))

                            else:
                                visitedStates = np.vstack((visitedStates, nextPossibleState)) #add the next possible state to the visited states list
                                nextStates.append(nextPossibleState)

                            #now that we have obtained an optimal control input based on the current node i; the next node j and the cost it took to get to it is an optimal cost-to-go from start to node j
                            #we add this key-value pair to the policy dictionary to further build our optimal which maps any state-time pair to a control input to build the shortest path to the goal

                            dictKey = np.zeros(28, dtype=np.int16)

                            #we will encode in the key of the key-value pair information that will allow us to trace back the policy from the goal node later
                            dictKey[0] = t #[0] is the time step
                            dictKey[1] = goalReached #[1] indicates if this key-value pair is a goal node that can be traced back
                            dictKey[2:15] = currentState #[2:14] is the current state node, the state that is to be mapped to a control input
                            dictKey[15:28] = nextPossibleState #[15:27] is the next state node that will result from the motion model with control input
                            policyDict[np.array2string(dictKey)[1:-1]] = controlInput #the value corresponding to this key is the control input
                            #at each time step, all the possible state transitions and their associated control inputs are stored in the policy dictionary, but only the optimal paths that reach the goal in the shortest time will get traced back and stored in a separate dictionary
            
            unprunedCurrentStates = np.array(nextStates) #we assume that the next possible states may not all be surviving if one branch has found the goal before everyone else
            currentStates = []

            if len(initialStatesWithGoalFound) > 1: #check if goals have been found for any of the initial states
                for unprunedCurrentState in unprunedCurrentStates:
                    if not checkIfStateBeenVisited(unprunedCurrentState[4:10], initialStatesWithGoalFound):
                        currentStates.append(unprunedCurrentState) #we only pass through branches that still have potential to find the shortest path for their initial states
                currentStates = np.array(currentStates, dtype=np.int16)
            else:
                currentStates = unprunedCurrentStates #all branches for this time step survive since no new goals have been found
            

        #at this point, the algorithm has terminated since all exploratory branches have been killed off and all initial states have found an optimal path to the goal
        #we now trace back the dictionary from the goal nodes to create the optimal policy for the random map
        optimalPolicy = {}


        for key, value in policyDict.items(): #iterate through all key-value pair
            key = np.fromstring(key, dtype=int, sep=' ')
            if key[1]: #if the pair represents finding the goal node
                for time in reversed(range(key[0]+1)): #trace back starting from the time at finding this goal node
                        currentState = key[15:28] #extract the state after the motion model
                        previousState = key[2:15] #extract the state before the motion model
                        controlInput = value #extract the control input to be fed into the motion model
                        if np.array2string(np.concatenate((np.array([time], dtype=np.int16),previousState)))[1:-1] not in optimalPolicy: #check if this state has not been mapped to a control input because there can be multiple optimal policies; we just choose in the order of the for loop
                            optimalPolicy[np.array2string(np.concatenate((np.array([time], dtype=np.int16),previousState)))[1:-1]] = controlInput
                            #we create a key-value pair where:
                            #key = [time step][state to be mapped to control input]
                            #value = [optimal control input]
                        for key_2, value_2 in policyDict.items():
                            key_2 = np.fromstring(key_2, dtype=int, sep=' ')
                            if key_2[0] == time - 1 and np.array_equal(key_2[15:28], previousState): #we now find the previous time step pair by checking the decremented time step AND if its next state is equal to our previous state
                                #change the current pair to the found one
                                key = key_2
                                value = value_2

        print("COMPLETED CALCULATING SINGLE OPTIMAL POLICY")                 
        return optimalPolicy #we now return an optimal policy dictionary where all possible state-time pairs can be mapped to an optimal control input to reach the goal node for any random map
