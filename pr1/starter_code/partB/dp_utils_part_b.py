import numpy as np

NONE = 0
DOOR = 1
KEY = 2
GOAL = 3
WALL = 4

#DOOR_1_POS = np.array([4,2], dtype=np.int16)
#DOOR_2_POS = np.array([4,5], dtype=np.int16)

DOOR_1_POS = np.array([1,1], dtype=np.int16) + np.array([1,1], dtype=np.int16)
DOOR_2_POS = np.array([1,2], dtype=np.int16) + np.array([1,1], dtype=np.int16)




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


def getNextPossibleStates(currentState, randomMap):

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