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

def getPreviousPossibleStates(currentState, env): #given a state vector node, what are the possible previous states?

    #the state cost from a possible state to the current state is simply 1, the cost of a valid control input
    #otherwise, the stage cost is infinite and the control input is not considered to be optimal for the value function at time t
    #for a given current state, go through all the 5 possible control inputs and determine which is valid and the inverse motion model output to get the previous state vector

    possibleStates = np.zeros((5,6), dtype=np.int16) #initialize an empty list to contain at most all the 5 previous state vectors that correspond to the 5 control inputs

    #extract information about the environment and the agent from the custom current state vector
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
    
    if (rearObject == "none"): #if there was nothing in the back, then a possible move would have been to move forward
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection #move forward inverse motion model is to move backwards
        possibleStates[0,:] = possibleState
    elif (rearObject == "door" and pickedUpKey and doorOpen): #if the door behind was open because we had the key, then we could have also moved forward from the door
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection
        possibleStates[0,:] = possibleState
    elif (rearObject == "key" and pickedUpKey): #if there used to be a key behind us that we now have in hand, we could have came from where the key was
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] - frontDirection
        possibleStates[0,:] = possibleState

    if (forwardObject == "door" and pickedUpKey and doorOpen): #if in front of us is an unlocked door and we have a key in hand, we could have previously done an unlock action
        possibleState = currentState.copy()
        possibleState[4] = 0 #change the door back to locked
        possibleStates[4,:] = possibleState
    elif (forwardObject == "key" and pickedUpKey): #if in front of us used to be a key and we now have that key in hand, we could have previously picked up the key
        possibleState = currentState.copy()
        possibleState[5] = 0 #go back to not having the key in hand
        possibleStates[3,:] = possibleState

    #for any state, it is always valid to turn left or right in place at a cost of 1
    turnRightState = currentState.copy()
    turnRightState[2:4] = leftDirection #inverse the turn right action by turning left
    possibleStates[2,:] = turnRightState

    turnLeftState = currentState.copy()
    turnLeftState[2:4] = rightDirection #inverse the turn left action by turning right
    possibleStates[1,:] = turnLeftState

    return possibleStates #once all the control inputs have been gone through, we return the ones that are valid and will incur a stage cost of 1
    

def checkIfStateBeenVisited(state, statesVisitedList):

    #compares a target state vector to a list of state vectors that have been visited
    #returns true if that target state vector exists in the list i.e. has been visited
    diff = np.abs(np.array(statesVisitedList) - state) #subtract target state from all states in list to find vector difference
    #if difference is exactly a zero vector then there is a match
    sum = np.sum(diff, axis = 1) #a zero vector's elements will sum to 0
    return not np.all(sum) #check if there is a sum of 0, and if so that means the target state has been visited


def calculateOptimalPolicy(env, info, timeHorizon):
    print("CALCULATING OPTIMAL SEQUENCE")
    #core of the dynamic programming algorithm

    goalPos = info["goal_pos"]

    #first we construct all the possible termination states that are desirable i.e. at the goal
    initialStates = []
    initialDirs = [np.array([0,1]), np.array([0,-1]), np.array([1,0]), np.array([-1,0])] #we can end at the goal from any of the 4 directions
    initialDoors = np.array([0,1]) #we could have gotten to the goal with or without unlocking the door
    initialKeys = np.array([0,1]) #similarly, with or without picking up the key

    for initialDir in initialDirs:
        for initialDoor in initialDoors:
            for initialKey in initialKeys:
                initialStates.append(createStateVector(goalPos, initialDir, initialDoor, initialKey)) #create the permutations

    currentStates = np.atleast_2d(initialStates)
    visitedStates = currentStates.copy() #initialize the visited nodes to only contain the initial node
    policyDict = {} #policy dictionary to map a state to a control input

    for t in range(timeHorizon): #we will iterate up to a time horizon
        
        nextStates = []
        for currentStateIndex in range(currentStates.shape[0]): #iterate through all the "surviving" nodes with a cost-to-go of t
            currentState = currentStates[currentStateIndex,:]
            nextPossibleStates = getPreviousPossibleStates(currentState, env) #find all the possible nodes before this node
            for controlInput in range(5): #iterate through the 5 possible control inputs
                nextPossibleState = nextPossibleStates[controlInput,:] #get the node that would use the corresponding control input to transition to the current node
                if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))): #check if the stage cost for that control input is not infinite i.e. if all zeros then impossible for that control input to optimally lead to current node
                    if not checkIfStateBeenVisited(nextPossibleState, visitedStates): #check if the state has been visited because then the cost-to-go is suboptimal i.e. node was already found at a previous t and has a dictionary entry
                        #at this point, the next node has been checked to satisfy both conditions 1. possible to transition from 2. has not been visited for the stage cost to be 1 and therefore optimal
                        visitedStates = np.vstack((visitedStates, nextPossibleState)) #add this next node to the visited states
                        nextStates.append(nextPossibleState) #add this next node to the next states list, which will be the future surviving current states
                        policyDict[np.array2string(nextPossibleState)[1:-1]] = controlInput

        currentStates = np.array(nextStates) #all surviving states have equal optimal cost-to-go values and therefore survive

        if (len(nextStates) == 0): #if every possible state has an optimal cost-to-go value found, and therefore a dictionary entry, then the complete policy has been found, so terminate the algorithm
            break

    return policyDict