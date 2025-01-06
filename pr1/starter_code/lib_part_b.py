import numpy as np
from tqdm import tqdm
 
#functions for part B

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

def getPreviousPossibleStates(currentState, randomMap, DOOR_1_POS): #given a state vector node, what are the possible previous states?

    #see comments for implementation in lib_part_a; it contains explantations of core logic behind creating the possible states
    #since part B involves a random map, we add logic to use the state vector to deduce what is actually in our surroundings vs. assuming for all cases

    pos = currentState[0:2]
    frontDirection = currentState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]])
    leftDirection = np.array([frontDirection[1],-frontDirection[0]])
    frontPos = pos + frontDirection
    rearPos = pos - frontDirection

    keyPos = currentState[6:8]
    doorState = currentState[8:10]
    keyPickedUp = currentState[10]
    door1UnlockAttempt = currentState[11]
    door2UnlockAttempt = currentState[12]

    #here we first assume that both the front and rear objects are NONE, but we go through logic based on the current state vector to deduce what the objects actually are
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

    #similar logic as above for deducing the rear object

    if (randomMap[rearPos[0], rearPos[1]] == DOOR):
        rearObject = DOOR
    if (randomMap[rearPos[0], rearPos[1]] == WALL):
        rearObject = WALL
    elif (np.array_equal(rearPos,keyPos)):
        rearObject = KEY

    #now the expected object front and rear has been deduced, we can perform the inverse motion model on the current state for all 5 control inputs just like in part A, with the addition of two possible doors with four possible initial states

    if (rearObject == NONE):
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection
        possibleStates[0,:] = possibleState
    elif (rearObject == DOOR): #if there was a door behind us
        if (np.array_equal(rearPos,DOOR_1_POS)): #check which door it is, is it door 1?
            if (doorState[0]): #check if door 1 is unlocked or locked initially, if unlocked
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] - frontDirection #then we could have moved forward from it
                possibleStates[0,:] = possibleState
            elif (keyPickedUp and door1UnlockAttempt): #if door 1 was locked initially, but we made an attempt at unlocking it, it probably is unlocked for us to move through
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] - frontDirection #then we could have moved forward from it
                possibleStates[0,:] = possibleState
        else: #similar logic for door 2
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


    if (frontObject == DOOR): #if there is a door in front of us
        if (np.array_equal(frontPos,DOOR_1_POS)): #is it door 1?
            if (not doorState[0] and door1UnlockAttempt): #if the door was initially locked and we did make an attempt at unlocking it
                possibleState = currentState.copy()
                possibleState[11] = 0 #then reverse that attempt so it has not been done yet
                possibleStates[4,:] = possibleState
        else: #similar logic for door 2
            if (not doorState[1] and door2UnlockAttempt):
                possibleState = currentState.copy()
                possibleState[12] = 0 
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
    #core of the dynamic programming algorithm, just like in part A

    randomMap = constructRandomMap(goalLocations, keyLocations, doorLocations, dimension) #create the probabilistic map based on the random generation parameters

    #first we construct all the possible termination states that are desirable i.e. at the goal positions
    #but now we also do permutations of the possible map configurations
    #we add [1,1] to the absolute cell coordinates due to the surrounding wall padding in the random map representation
    initialGoalLocations = np.array(goalLocations, dtype=np.int16) + np.array([1,1])
    initialKeyLocations = np.array(keyLocations, dtype=np.int16) + np.array([1,1])
    initialDirs = [np.array([0,1]), np.array([0,-1]), np.array([1,0]), np.array([-1,0])]
    initialDoorStates = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]

    #IMPORTANT EFFICIENCY BOOSTER
    #we can increase run time before not considering both doors to be attempted at unlock by the time we reach the goal
    #this is realistic because why would the agent need to unlock both doors?
    #this also means technically the policy is *NOT* fully complete, but for the purposes of this project and time, it still does the job very well
    initialDoorUnlockAttemptedStates = [np.array([0,0]), np.array([0,1]), np.array([1,0])] 


    initialKeyPickedUpStates = np.array([0,1])

    initialStates = []

    for initialGoalLocation in initialGoalLocations:
        for initialKeyLocation in initialKeyLocations:
            for initialDir in initialDirs:
                for initialDoorState in initialDoorStates:
                    for initialDoorUnlockAttemptedState in initialDoorUnlockAttemptedStates:
                        for initialKeyPickedUpState in initialKeyPickedUpStates:
                            initialStates.append(createStateVector(initialGoalLocation, initialDir, initialGoalLocation, initialKeyLocation, initialDoorState, initialKeyPickedUpState, initialDoorUnlockAttemptedState[0], initialDoorUnlockAttemptedState[1]))


    currentStates = np.atleast_2d(initialStates)
    visitedStates = currentStates.copy() #initialize the visited nodes to only contain the initial node
    policyDict = {}

    DOOR_1_POS = doorLocations[0] + np.array([1,1])
    #DOOR_2_POS = doorLocations[1] + np.array([1,1])

    for t in tqdm (range (timeHorizon), desc="time horizon step"): #we will iterate up to a time horizon
        
        nextStates = []
        for currentStateIndex in tqdm (range (currentStates.shape[0]), desc="current state"): #iterate through all the "surviving" nodes with a cost-to-go of t
            currentState = currentStates[currentStateIndex,:]
            nextPossibleStates = getPreviousPossibleStates(currentState, randomMap, DOOR_1_POS) #find all the possible nodes before this node
            for controlInput in range(5): #iterate through the 5 possible control inputs
                nextPossibleState = nextPossibleStates[controlInput,:] #get the node that would use the corresponding control input to transition to the current node
                if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))): #check if the stage cost for that control input is not infinite i.e. if all zeros then impossible for that control input to optimally lead to current node
                    if not checkIfStateBeenVisited(nextPossibleState, visitedStates): #check if the state has been visited because then the cost-to-go is suboptimal i.e. node was already found at a previous t and has a dictionary entry
                        #at this point, the next node has been checked to satisfy both conditions 1. possible to transition from 2. has not been visited for the stage cost to be 1 and therefore optimal
                        visitedStates = np.vstack((visitedStates, nextPossibleState)) #add this next node to the visited states
                        nextStates.append(nextPossibleState) #add this next node to the next states list, which will be the future surviving current states
                        offsetNextPossibleState = nextPossibleState - np.array([1,1,0,0,1,1,1,1,0,0,0,0,0]) #correct offset of wall padding
                        policyDict[np.array2string(offsetNextPossibleState)[1:-1]] = controlInput

        currentStates = np.array(nextStates) #all surviving states have equal optimal cost-to-go values and therefore survive

        if (len(nextStates) == 0): #if every possible state has an optimal cost-to-go value found, and therefore a dictionary entry, then the complete policy has been found, so terminate the algorithm
            break

    return policyDict