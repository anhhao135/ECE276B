import numpy as np

NONE = 0
DOOR = 1
KEY = 2
GOAL = 3
WALL = 4


def constructRandomMap(goalLocations, keyLocations, doorLocations, dimension):
    randomMap = np.zeros((dimension,dimension), dtype=np.int16)
    wallColumn = doorLocations[0][0]

    randomMap[wallColumn,:] = WALL

    for goalLocation in goalLocations:
        randomMap[goalLocation[0], goalLocation[1]] = GOAL

    for keyLocation in keyLocations:
        randomMap[keyLocation[0], keyLocation[1]] = KEY

    for doorLocation in doorLocations:
        randomMap[doorLocation[0], doorLocation[1]] = DOOR

    surroundingWallsMap = np.full((dimension+2,dimension+2), WALL, dtype=np.int16)
    surroundingWallsMap[1:dimension+1, 1:dimension+1] = randomMap

    return surroundingWallsMap

def createStateVector(pos, dir, goal, key, door, keyPickedUp):
    state = np.zeros((1,13), dtype=np.int16).flatten()
    state[0:2] = pos
    state[2:4] = dir
    state[4:6] = goal
    state[6:8] = key
    state[8:10] = door
    state[10] = keyPickedUp
    state[10:12] = np.zeros(2,dtype=np.int16)
    return state

def createCurrentStateVector(currentPos, currentDir, goalPos, keyPos, doorState, currentKeyPickedUp, door1UnlockAttempted, door2UnlockAttempted):
    state = np.zeros((1,13), dtype=np.int16).flatten()
    state[0:2] = currentPos
    state[2:4] = currentDir
    state[4:6] = goalPos
    state[6:8] = keyPos
    state[8:10] = doorState
    state[10] = currentKeyPickedUp
    state[11] = door1UnlockAttempted
    state[12] = door2UnlockAttempted
    return state



def createEndGoalStates(goalState):
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


def getNextPossibleStates(currentState, randomMap, DOOR_1_POS):

    #print("current state is:")
    #print(currentState)



    pos = currentState[0:2]
    frontDirection = currentState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]])
    leftDirection = np.array([frontDirection[1],-frontDirection[0]])

    frontPos = pos + frontDirection

    goalPos = currentState[4:6]
    keyPos = currentState[6:8]
    doorState = currentState[8:10]
    keyPickedUp = currentState[10]
    door1UnlockAttempt = currentState[11]
    door2UnlockAttempt = currentState[12]

    frontObject = NONE

    possibleStates = np.zeros((6,13), dtype=np.int16)

    if (np.array_equal(pos,goalPos)): #stay
        print("goal found!!!")
        possibleState = currentState.copy()
        possibleStates[5,:] = possibleState
        return possibleStates

    if (np.array_equal(frontPos,goalPos)):
        frontObject = GOAL
    elif (np.array_equal(frontPos,keyPos) and not keyPickedUp):
        frontObject = KEY
    elif (randomMap[frontPos[0], frontPos[1]] == DOOR):
        if (np.array_equal(frontPos,DOOR_1_POS)):
            if (doorState[0] or door1UnlockAttempt): #top door is unlocked
                frontObject = NONE
            else:
                frontObject = DOOR
        else:
            if (doorState[1] or door2UnlockAttempt): #bottom door is unlocked
                frontObject = NONE
            else:
                frontObject = DOOR
    elif (randomMap[frontPos[0], frontPos[1]] == WALL):
        frontObject = WALL


    if (frontObject == NONE or frontObject == GOAL):
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] + frontDirection
        possibleStates[0,:] = possibleState
    elif (frontObject == KEY):
        possibleState = currentState.copy()
        possibleState[10] = 1
        possibleStates[3,:] = possibleState
    elif (frontObject == DOOR and keyPickedUp):
        if (np.array_equal(frontPos,DOOR_1_POS)): #unlock top door
            possibleState = currentState.copy()
            possibleState[11] = 1
            possibleStates[4,:] = possibleState
        else: #unlock bottom door
            possibleState = currentState.copy()
            possibleState[12] = 1
            possibleStates[4,:] = possibleState

    turnRightState = currentState.copy()
    turnRightState[2:4] = rightDirection
    possibleStates[2,:] = turnRightState

    turnLeftState = currentState.copy()
    turnLeftState[2:4] = leftDirection
    possibleStates[1,:] = turnLeftState

    return possibleStates

def checkIfStateBeenVisited(state, statesVisitedList):
    diff = np.abs(statesVisitedList - state)
    sum = np.sum(diff, axis = 1)
    return not np.all(sum)


def calculateSingleOptimalPolicy(timeHorizon, goalLocations, keyLocations, doorLocations, dimension, initialPos, initialDir):

    initialPos = initialPos + np.array([1,1])

    DOOR_1_POS = doorLocations[0] + np.array([1,1])
    DOOR_2_POS = doorLocations[1] + np.array([1,1])
    
    randomMap = constructRandomMap(goalLocations, keyLocations, doorLocations, dimension)

    goalLocations = np.array(goalLocations, dtype=np.int16) + np.array([1,1])
    keyLocations = np.array(keyLocations, dtype=np.int16) + np.array([1,1])
    doorStates = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.int16)

    initialStates =  []

    for goalLocation in goalLocations:
        for keyLocation in keyLocations:
            for doorState in doorStates:
                initialState = createStateVector(initialPos, initialDir, goalLocation, keyLocation, doorState, 0)
                initialStates.append(initialState)

    currentStates = np.atleast_2d(np.array(initialStates))
    print(currentStates.shape)
    visitedStates = currentStates.copy()
    print("initial visited states")
    print(visitedStates)

    policyDict = {}

    for t in range(timeHorizon):
        print("------------------------")
        print(t)
        nextStates = []
        currentControlInputs = []
        goalControlInput = None

        initialStatesWithGoalFound = [np.array([-1,-1,-1,-1,-1,-1])]

        for currentStateIndex in range(currentStates.shape[0]):
            currentState = currentStates[currentStateIndex,:]
            nextPossibleStates = getNextPossibleStates(currentState, randomMap, DOOR_1_POS)
            for controlInput in range(6): #the 6 different control inputs
                nextPossibleState = nextPossibleStates[controlInput,:]
                if not (np.array_equal(nextPossibleState, np.zeros(13, dtype=np.int16))):
                    if not checkIfStateBeenVisited(nextPossibleState, visitedStates):
                        print("next possible state")
                        print(nextPossibleState)
                        goalReached = np.array_equal(nextPossibleState[0:2], nextPossibleState[4:6])
                        if (goalReached):
                            print("goal reached")
                            if not checkIfStateBeenVisited(nextPossibleState[4:10], initialStatesWithGoalFound):
                                initialStatesWithGoalFound.append(nextPossibleState[4:10])
                            endGoalStates = createEndGoalStates(nextPossibleState)
                            visitedStates = np.vstack((visitedStates, endGoalStates))

                        else:
                            visitedStates = np.vstack((visitedStates, nextPossibleState))
                            nextStates.append(nextPossibleState)
                        dictKey = np.zeros(28, dtype=np.int16)
                        dictKey[0] = t
                        dictKey[1] = goalReached
                        dictKey[2:15] = currentState
                        dictKey[15:28] = nextPossibleState
                        policyDict[np.array2string(dictKey)[1:-1]] = controlInput
        
        unprunedCurrentStates = np.array(nextStates)
        currentStates = []
        #controlInputs.append(currentControlInputs)

        if len(initialStatesWithGoalFound) > 1:
            for unprunedCurrentState in unprunedCurrentStates:
                if not checkIfStateBeenVisited(unprunedCurrentState[4:10], initialStatesWithGoalFound):
                    currentStates.append(unprunedCurrentState)
            currentStates = np.array(currentStates, dtype=np.int16)
        else:
            currentStates = unprunedCurrentStates


        print("current states")
        print(currentStates)
        print("visited states")
        print(visitedStates)
        print("initial states with goal found")
        print(initialStatesWithGoalFound)
        print("------------------------")
        
    #print(policyDict)

    optimalPolicy = {}


    for key, value in policyDict.items():
        key = np.fromstring(key, dtype=int, sep=' ')
        if key[1]:
            print(key[0])
            for time in reversed(range(key[0]+1)):
                    print("here")
                    currentState = key[15:28]
                    previousState = key[2:15]
                    controlInput = value
                    if np.array2string(np.concatenate((np.array([time], dtype=np.int16),previousState)))[1:-1] not in optimalPolicy:
                        optimalPolicy[np.array2string(np.concatenate((np.array([time], dtype=np.int16),previousState)))[1:-1]] = controlInput
                    for key_2, value_2 in policyDict.items():
                        key_2 = np.fromstring(key_2, dtype=int, sep=' ')
                
                        if key_2[0] == time - 1 and np.array_equal(key_2[15:28], previousState):
                            print("000000000000000000")
                            print(key_2[15:28])
                            print(value_2)
                            print(previousState)
                            print("000000000000000000")
                            key = key_2
                            value = value_2
                        
    return optimalPolicy
