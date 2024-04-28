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
    
def getCurrentState(env, envInfo): #construct a custom state vector based on the state of the environment
    agentPosition = env.agent_pos
    agentDirection = env.dir_vec
    door = env.grid.get(envInfo["door_pos"][0], envInfo["door_pos"][1])
    doorIsOpenStatus = door.is_open
    keyPickedUpStatus = env.carrying is not None
    stateVector = np.array([agentPosition[0],agentPosition[1],agentDirection[0],agentDirection[1],int(doorIsOpenStatus),int(keyPickedUpStatus)])
    #stateVector is a custom state defined as follows:
    # agent position:[0:1]; agent direction:[2:3]; door unlocked?:[4]; key picked up?:[5]
    return stateVector

def getNextPossibleStates(currentState, env): #given a state vector node, what are the possible next states?
    #the state cost to a possible state is simply 1, the cost of a valid control input
    #otherwise, the stage cost is infinite and the control input is not considered to be optimal for the value function at time T
    #for a given current state, go through all the 5 possible control inputs and determine which is valid and the motion model output to get the next state vector

    possibleStates = np.zeros((5,6), dtype=np.int16) #initialize an empty list to contain at most all the 5 next valid state vectors that correspond to the 5 control inputs

    #extract information about the environment and the agent from the custom current state vector
    pos = currentState[0:2]
    frontDirection = currentState[2:4]
    rightDirection = np.array([-frontDirection[1],frontDirection[0]]) #the right direction unit vector, given the front is [x,y]. is [-y,x]
    leftDirection = np.array([frontDirection[1],-frontDirection[0]]) #the left direction unit vector, given the front is [x,y]. is [y,-x]
    doorOpen = currentState[4]
    pickedUpKey = currentState[5]
    
    #get the surround object types to determine the valid control inputs
    #add the direction vectors to the current position to get the cell coordinate one step in that direction
    forwardObject = getTypeAtCell(env, pos + frontDirection)
    rightObject = getTypeAtCell(env, pos + rightDirection)
    leftObject = getTypeAtCell(env, pos + leftDirection)

    if (forwardObject == "wall" and rightObject == "wall" and leftObject == "wall"):
        #first thing is to check if the agent is at a dead end i.e. surrounded by walls except for the rear
        print("DEAD END")
        #if so, then the path is sub-optimal since it has to do a u-turn to get back on the optimal path i.e. it took a detour
        #so return the empty possible states list to signify that every control input will yield an infinite stage cost to discourage exploring this path further
        return possibleStates
    else:
        if (forwardObject == "goal"): #if a goal as been found in front, then the next sensible action and most optimal is to move forward
            print("GOAL FOUND")
            return "GOAL FOUND" #return this string so the dynamic programming algorithm terminates
        if (forwardObject == "none"): #if there is nothing in the front, then a valid action would be to move forward
            nextPos = pos + frontDirection #motion model to move forward is to add the front direction vector to the current position
            possibleState = np.concatenate((nextPos, currentState[2:]), axis=None)
            possibleStates[0,:] = possibleState #add the move forward motion model output state vector to the list, at the 0th index to correspond to MF = 0
        elif (forwardObject == "key"): #if there is a key ahead, then we have to check if it has been picked up, or else it could be blocking us
            if (pickedUpKey): #if the key has been previously picked up, we know effectively there is nothing ahead of us, so this is equivalent to a MF control input
                possibleState = currentState.copy()
                possibleState[0:2] = possibleState[0:2] + frontDirection #move forward motion model
                possibleStates[0,:] = possibleState
            else:
                possibleState = currentState.copy() #if the key has not been picked up, it is blocking our way, so the next valid control input would be to pick it up, PK
                possibleState[5] = 1 #the next state would have the picked up key value to 1 following the motion model
                possibleStates[3,:] = possibleState
        elif (forwardObject == "door" and pickedUpKey): #if there is a door ahead but we don't have a key in hand, there is no optimal next move 
            #we only consider a next move if there is a door and we have a key in hand because that could mean two things:
            if (doorOpen): #check if that door has been previously unlocked, if so then this is equivalent to not having anything in front of us so we can move forward
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
    

def checkIfStateBeenVisited(state, statesVisitedList):
    #compares a target state vector to a list of state vectors that have been visited
    #returns true if that target state vector exists in the list i.e. has been visited
    diff = np.abs(np.array(statesVisitedList) - state) #subtract target state from all states in list to find vector difference
    #if difference is exactly a zero vector then there is a match
    sum = np.sum(diff, axis = 1) #a zero vector's elements will sum to 0
    return not np.all(sum) #check if there is a sum of 0, and if so that means the target state has been visited

def traceBackPolicy(controlInputs):
    policySequence = []
    timeSteps = len(controlInputs)
    previousControlIndex = 0
    for i in range(timeSteps-1, -1, -1):
        step = controlInputs[i][previousControlIndex]
        previousControlIndex = step[0]
        policySequence.insert(0, step[1])
    return policySequence

def calculateOptimalSequence(env, info, timeHorizon):

    currentStates = np.atleast_2d(getCurrentState(env,info))
    visitedStates = currentStates.copy()
    print("initial visited states")
    print(visitedStates)
    timeHorizon = 1000
    controlInputs = []
    flagGoalFound = False


    for t in range(timeHorizon):
        print("------------------------")
        print(t)
        nextStates = []
        currentControlInputs = []
        goalControlInput = None
        for currentStateIndex in range(currentStates.shape[0]):
            currentState = currentStates[currentStateIndex,:]
            nextPossibleStates = getNextPossibleStates(currentState, env)
            if nextPossibleStates == "GOAL FOUND":
                flagGoalFound = True
                goalControlInput = np.array([currentStateIndex, 0]) #only way is to move forward
                break
            else:
                for controlInput in range(5): #the 5 different control inputs
                    nextPossibleState = nextPossibleStates[controlInput,:]
                    if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))):
                        if not checkIfStateBeenVisited(nextPossibleState, visitedStates):
                            print("next possible state")
                            print(nextPossibleState)
                            visitedStates = np.vstack((visitedStates, nextPossibleState))
                            nextStates.append(nextPossibleState)
                            currentControlInputs.append(np.array([currentStateIndex, controlInput]))
        if flagGoalFound:
            controlInputs.append([goalControlInput])
            break
        currentStates = np.array(nextStates)
        controlInputs.append(currentControlInputs)
        print("current states")
        print(currentStates)
        print("visited states")
        print(visitedStates)
        print("------------------------")
        

    print(controlInputs)
    seq = traceBackPolicy(controlInputs)
    return seq