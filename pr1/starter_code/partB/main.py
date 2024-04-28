from dp_utils_part_b import *

#goalLocations = [np.array([5,1]), np.array([6,3]), np.array([5,6])]
#keyLocations = [np.array([1,1]), np.array([2,3]), np.array([1,6])]
#doorLocations = [np.array([4,2]), np.array([4,5])]
#dimension = 8

goalLocations = [np.array([2,0]), np.array([2,1])]
keyLocations = [np.array([0,0]), np.array([0,1])]
doorLocations = [np.array([1,1]), np.array([1,2])]
dimension = 3

initialPos = np.array([0,2]) + np.array([1,1])
initialDir = np.array([0,-1])
initialGoal = np.array([2,0]) + np.array([1,1])
initialKey = np.array([0,1]) + np.array([1,1])
initialDoor = np.array([0,0]) #top and bottom one unlocked locked
initialKeyPickedUp = 0

initialState1 = createStateVector(initialPos, initialDir, initialGoal, initialKey, initialDoor, initialKeyPickedUp)

initialPos = np.array([0,2]) + np.array([1,1])
initialDir = np.array([0,-1])
initialGoal = np.array([2,0]) + np.array([1,1])
initialKey = np.array([0,1]) + np.array([1,1])
initialDoor = np.array([1,1]) #top one unlocked and bottom one locked
initialKeyPickedUp = 0

initialState2 = createStateVector(initialPos, initialDir, initialGoal, initialKey, initialDoor, initialKeyPickedUp)

initialPos = np.array([0,2]) + np.array([1,1])
initialDir = np.array([0,-1])
initialGoal = np.array([2,0]) + np.array([1,1])
initialKey = np.array([0,1]) + np.array([1,1])
initialDoor = np.array([1,0]) #top one unlocked and bottom one locked
initialKeyPickedUp = 0

initialState3 = createStateVector(initialPos, initialDir, initialGoal, initialKey, initialDoor, initialKeyPickedUp)

initialState = [initialState1, initialState2, initialState3]
#initialState = [initialState3]

policyDict = {}





randomMap = constructRandomMap(goalLocations, keyLocations, doorLocations, dimension)
print(randomMap.T)

currentStates = np.atleast_2d(np.array(initialState))
print(currentStates.shape)
visitedStates = currentStates.copy()
print("initial visited states")
print(visitedStates)
timeHorizon = 15
controlInputs = []
flagGoalFound = False

for t in range(timeHorizon):
    print("------------------------")
    print(t)
    nextStates = []
    currentControlInputs = []
    goalControlInput = None

    initialStatesWithGoalFound = [np.array([-1,-1,-1,-1,-1,-1])]

    for currentStateIndex in range(currentStates.shape[0]):
        currentState = currentStates[currentStateIndex,:]
        nextPossibleStates = getNextPossibleStates(currentState, randomMap)
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
                
print(optimalPolicy)