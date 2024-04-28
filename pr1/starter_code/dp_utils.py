from dp_utils import *
from minigrid.core.world_object import Goal, Key, Door, Wall
import numpy as np

def getObjectType(object):
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

def getTypeAtCell(env, pos):
    gotObject = env.grid.get(pos[0],pos[1])
    return getObjectType(gotObject)

def getTypeInFront(env):
    frontObject = env.grid.get(env.front_pos[0],env.front_pos[1])
    return getObjectType(frontObject)


def getTypeLeft(env):
    forwardDirectionVector = env.dir_vec
    leftDirectionVector = np.array([forwardDirectionVector[1],-forwardDirectionVector[0]])
    leftCell = env.agent_pos + leftDirectionVector
    leftObject = env.grid.get(leftCell[0],leftCell[1])
    return getObjectType(leftObject)

def getTypeRight(env):
    forwardDirectionVector = env.dir_vec
    rightDirectionVector = np.array([-forwardDirectionVector[1],forwardDirectionVector[0]])
    rightCell = env.agent_pos + rightDirectionVector
    rightObject = env.grid.get(rightCell[0],rightCell[1])
    return getObjectType(rightObject)

def checkIfDeadEnd(env):
    if (getTypeInFront(env) == "wall" and getTypeRight(env) == "wall" and getTypeLeft(env) == "wall"):
        return True
    else:
        return False
    
def getCurrentState(env, envInfo):
    agentPosition = env.agent_pos
    agentDirection = env.dir_vec
    door = env.grid.get(envInfo["door_pos"][0], envInfo["door_pos"][1])
    doorIsOpenStatus = door.is_open
    keyPickedUpStatus = env.carrying is not None
    stateVector = np.array([agentPosition[0],agentPosition[1],agentDirection[0],agentDirection[1],int(doorIsOpenStatus),int(keyPickedUpStatus)])
    return stateVector

def getNextPossibleStates(currentState, env):

    #print("current state is:")
    #print(currentState)

    possibleStates = np.zeros((5,6), dtype=np.int16)

    pos = currentState[0:2]
    frontDirection = currentState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]])
    leftDirection = np.array([frontDirection[1],-frontDirection[0]])
    doorOpen = currentState[4]
    pickedUpKey = currentState[5]
    
    forwardObject = getTypeAtCell(env, pos + frontDirection)
    rightObject = getTypeAtCell(env, pos + rightDirection)
    leftObject = getTypeAtCell(env, pos + leftDirection)

    if (forwardObject == "wall" and rightObject == "wall" and leftObject == "wall"):
        print("dead end")
        return possibleStates
    else:
        if (forwardObject == "goal"):
            print("GOAL FOUND")
            return "GOAL FOUND"
        if (forwardObject == "none"):
            nextPos = pos + frontDirection
            possibleState = np.concatenate((nextPos, currentState[2:]), axis=None)
            possibleStates[0,:] = possibleState
        elif (forwardObject == "key"):
            if (pickedUpKey): #move forward
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] + frontDirection
                possibleStates[0,:] = possibleState
            else:
                possibleState = currentState.copy() #pickup key
                possibleState[5] = 1
                possibleStates[3,:] = possibleState
        elif (forwardObject == "door" and pickedUpKey):
            if (doorOpen): #move forward
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] + frontDirection
                possibleStates[0,:] = possibleState
            else: #unlock door
                possibleState = currentState.copy()
                possibleState[4] = 1
                possibleStates[4,:] = possibleState

        turnRightState = currentState.copy()
        turnRightState[2:4] = rightDirection
        possibleStates[2,:] = turnRightState

        turnLeftState = currentState.copy()
        turnLeftState[2:4] = leftDirection
        possibleStates[1,:] = turnLeftState


        return possibleStates
    

def filterOutUnpromisingNextPossibleStates(nextPossibleStates, globalStatesVisitedList, env):
    for i in range(5):
        if not np.array_equal(nextPossibleStates[i,:], np.zeros(6)):
            print(i)
            nextNextPossibleStates = getNextPossibleStates(nextPossibleStates[i,:], env)
            print(nextNextPossibleStates)


def checkIfNextStateIsPromising(nextState, globalStatesVisitedList, env):
    pos = nextState[0:2]
    frontDirection = nextState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]])
    leftDirection = np.array([frontDirection[1],-frontDirection[0]])
    doorOpen = nextState[4]
    pickedUpKey = nextState[5]

def checkIfStateBeenVisited(state, statesVisitedList):
    diff = np.abs(np.array(statesVisitedList) - state)
    sum = np.sum(diff, axis = 1)
    return not np.all(sum)

def traceBackPolicy(controlInputs):
    policySequence = []
    timeSteps = len(controlInputs)
    previousControlIndex = 0
    for i in range(timeSteps-1, -1, -1):
        step = controlInputs[i][previousControlIndex]
        previousControlIndex = step[0]
        policySequence.insert(0, step[1])
    return policySequence