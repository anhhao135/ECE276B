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

def createStateVector(pos, dir, door, key): #here the custom state vector format is defined
    state = np.zeros((1,6), dtype=np.int16).flatten()
    state[0:2] = pos
    state[2:4] = dir
    state[4] = door
    state[5] = key
    return state


def getNextPossibleStates(currentState, env, policyDict): #given a state vector node, what are the possible next states?
    #the state cost to a possible state is simply 1, the cost of a valid control input
    #otherwise, the stage cost is infinite and the control input is not considered to be optimal for the value function at time t
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


    if (forwardObject == "goal"): #if a goal as been found in front, then the next sensible action and most optimal is to move forward
        possibleState = currentState.copy()
        possibleState[0:2] = possibleState[0:2] + frontDirection #move forward motion model
        possibleStates[0,:] = possibleState
        print("GOAL FOUND")
        return possibleStates, False, -2 #return this string so the dynamic programming algorithm terminates
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
            possibleState[0:2] = possibleState[0:2] + frontDirection #move forward motion model
            possibleStates[0,:] = possibleState
        else: #if the door is locked, then we can proceed with unlocking it
            possibleState = currentState.copy()
            possibleState[4] = 1 #motion model for an unlock door move is to change the vector state portion to unlock
            possibleStates[4,:] = possibleState

    #for any state, it is always valid to turn left or right in place at a cost of 1
    turnRightState = currentState.copy()
    turnRightState[2:4] = rightDirection #motion model is to modify the agent direction
    possibleStates[2,:] = turnRightState

    turnLeftState = currentState.copy()
    turnLeftState[2:4] = leftDirection
    possibleStates[1,:] = turnLeftState

    valueFunctions = np.ones(5, dtype=np.int16) * 1000

    for i in range(5):
        if np.array2string(possibleStates[i,:])[1:-1] in policyDict:
            print("wyuwuwuwu")
            valueFunctions[i] = policyDict[np.array2string(possibleStates[i,:])[1:-1]][1]
            print(policyDict[np.array2string(possibleStates[i,:])[1:-1]][1])
            print(np.argmin(valueFunctions))

    if not np.array_equal(valueFunctions, np.ones(5, dtype=np.int16) * 1000):
        possibleStatesCopy = np.zeros((5,6), dtype=np.int16)
        possibleStatesCopy[np.argmin(valueFunctions),:] = possibleStates[np.argmin(valueFunctions),:]
        print("heer333")
        print(policyDict[np.array2string(possibleStates[np.argmin(valueFunctions),:])[1:-1]][1])
        return possibleStatesCopy, True, policyDict[np.array2string(possibleStates[np.argmin(valueFunctions),:])[1:-1]][1]

    return possibleStates, False, -1 #once all the control inputs have been gone through, we return the ones that are valid and will incur a stage cost of 1
    

def checkIfStateBeenVisited(state, statesVisitedList):
    #compares a target state vector to a list of state vectors that have been visited
    #returns true if that target state vector exists in the list i.e. has been visited
    diff = np.abs(np.array(statesVisitedList) - state) #subtract target state from all states in list to find vector difference
    #if difference is exactly a zero vector then there is a match
    sum = np.sum(diff, axis = 1) #a zero vector's elements will sum to 0
    return not np.all(sum) #check if there is a sum of 0, and if so that means the target state has been visited

def traceBackPolicy(controlInputs, policyDict, timeSteps = -1):

    #the transition chain is as follows
    # t:  ....     N-1                                N
    # chain:     [[1, TR],[2, PK],....]           [[0, MF]]
    # each time step is a list of [previousIndex, controlInput] pairs
    # previousIndex points to the index of the previous pair in the previous time

    policySequence = [] #initiate an empty optimal sequence chain to be later constructed as we traverse the chain
    if timeSteps <= -1:
        timeSteps = len(controlInputs) #this represents the optimal cost-to-go from start to goal node
    else:
        timeSteps = timeSteps + len(controlInputs)
    previousControlIndex = 0 #for the goal step, we know the previous pair is in the 0th index
    for i in range(len(controlInputs)-1, -1, -1): #iterate starting from end time to 0
        step = controlInputs[i][previousControlIndex] #extract the previous pair
        print(controlInputs[i])
        print("----------------------------")
        previousControlIndex = step[0] #we know the next previous pair index can be taken from the 1st element of the pair
        policySequence.insert(0, step[1]) #we can add to the front of the optimal sequence the control input, taken from the 2nd element of the pair
        policyDict[np.array2string(step[2:8])[1:-1]] = np.array([step[1], timeSteps - i])

    return policySequence, policyDict

def calculateOptimalSequence(env, info, timeHorizon, possiblePositions):

    goalPos = info["goal_pos"]
    print(goalPos)

    print("CALCULATING OPTIMAL SEQUENCE")
    #core of the forward dynamic programming algorithm

    policyDict = {}

    possibleDirections = [np.array([0,-1],dtype=np.int16), np.array([-1,0],dtype=np.int16), np.array([1,0],dtype=np.int16), np.array([0,1],dtype=np.int16)]

    initialStates = []

    for possiblePosition in possiblePositions:
        for possibleDirection in possibleDirections:
            initialStates.append(createStateVector(possiblePosition, possibleDirection, 0, 0))

    #initialStates = [createStateVector(np.array([1,2], dtype=np.int16), np.array([0,1], dtype=np.int16), 0, 0)]

    for times in range(len(initialStates)):

        print("here jordan")

        timeSteps = -1

        #currentStates = np.atleast_2d(getCurrentState(env,info)) #start by initializing the current state node
        currentStates = np.atleast_2d(initialStates[times]) #start by initializing the current state node
        visitedStates = currentStates.copy() #initialize the visited nodes to only contain the initial node
        controlInputs = [] #keep track of the control inputs that induces transitions between nodes
        flagGoalFound = False #keep track of whether the goal has been found; if so, the algorithm can terminate

        for t in range(timeHorizon): #we will iterate up to a time horizon, and assume that the optimal cost-to-arrive at the current nodes is equal to t
            nextStates = []
            currentControlInputs = []
            goalControlInput = None
            #print(t)
            #print(currentStates)

            for currentStateIndex in range(currentStates.shape[0]): #iterate through all the "surviving" nodes with a cost-to-arrive of t; these still have potential to be part of the shortest path
                currentState = currentStates[currentStateIndex,:]
                optimalPathFromOtherBranch = False
                nextPossibleStates, optimalPathFromOtherBranch, timeSteps = getNextPossibleStates(currentState, env, policyDict) #for each surviving node, find all the potential next nodes with corresponding control inputs
                #print(nextPossibleStates)
                if timeSteps == -2: #if a next node is a goal node, then terminate the algorithm, and the cost-to-arrive at the goal is t
                    flagGoalFound = True #set the goal found tracking flag
                    goalControlInput = np.concatenate((np.array([currentStateIndex, 0]), currentStates[currentStateIndex]))

                    print("here!!!")
                    print(goalControlInput)
                    break
                elif optimalPathFromOtherBranch:
                    
                    flagGoalFound = True #set the goal found tracking flag
                    for controlInput in range(5): #iterate through the 5 possible control inputs
                        nextPossibleState = nextPossibleStates[controlInput,:] #get the next node corresponding to that control input
                        if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))): #check if the stage cost for that control input is not infinite
                            if not checkIfStateBeenVisited(nextPossibleState, visitedStates) or optimalPathFromOtherBranch: #check if the state has been visited because then the cost-to-arrive is not optimal
                                #at this point, the next node has been checked to satisfy both conditions such that the stage cost is 1, or else it would be infinite and not considered
                                visitedStates = np.vstack((visitedStates, nextPossibleState)) #add this next node to the visited states
                                nextStates.append(nextPossibleState) #add this next node to the next states list, which will be the future surviving current states
                                currentControlInputs.append(np.concatenate((np.array([currentStateIndex, controlInput]), currentStates[currentStateIndex]))) #add the optimal control input that is associated with this state
                    goalControlInput = currentControlInputs
                    break
                else:
                    for controlInput in range(5): #iterate through the 5 possible control inputs
                        nextPossibleState = nextPossibleStates[controlInput,:] #get the next node corresponding to that control input
                        if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))): #check if the stage cost for that control input is not infinite
                            if not checkIfStateBeenVisited(nextPossibleState, visitedStates) or optimalPathFromOtherBranch: #check if the state has been visited because then the cost-to-arrive is not optimal
                                #at this point, the next node has been checked to satisfy both conditions such that the stage cost is 1, or else it would be infinite and not considered
                                visitedStates = np.vstack((visitedStates, nextPossibleState)) #add this next node to the visited states
                                nextStates.append(nextPossibleState) #add this next node to the next states list, which will be the future surviving current states
                                currentControlInputs.append(np.concatenate((np.array([currentStateIndex, controlInput]), currentStates[currentStateIndex]))) #add the optimal control input that is associated with this state

            if flagGoalFound:
                controlInputs.append([goalControlInput])
                break
            currentStates = np.array(nextStates)
            controlInputs.append(currentControlInputs) #add all the current step's possible transitions, and the associated control inputs, to the node transition chain
            
        print(controlInputs)
        seq, policyDict = traceBackPolicy(controlInputs, policyDict, timeSteps) #trace back the optimal chain starting from the last goal node
        #print("FOUND OPTIMAL SEQUENCE")
        #print(policyDict)
    return policyDict