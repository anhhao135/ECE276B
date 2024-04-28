from lib_part_b import *
from utils import *
from example import example_use_of_gym_env
from minigrid.core.world_object import Goal, Key, Door, Wall
import time
import random


goalLocations = [np.array([5,1]), np.array([6,3]), np.array([5,6])]
keyLocations = [np.array([1,1]), np.array([2,3]), np.array([1,6])]
doorLocations = [np.array([4,2]), np.array([4,5])]
dimension = 8


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



env, info = load_env("./envs/random_envs/doorkey-8x8-1.env")

# Visualize the environment
plot_env(env)

while False:

    # Get the agent position
    agent_pos = env.agent_pos

    # Get the agent direction
    agent_dir = env.dir_vec  # or env.agent_dir

    # Get the cell in front of the agent
    front_cell = env.front_pos  # == agent_pos + agent_dir

    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3)  # NoneType, Wall, Key, Goal

    # Get the door status
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
    is_open = door.is_open
    is_locked = door.is_locked

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None

    # Take actions
    cost, done = step(env, MF)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Moving Forward Costs: {}".format(cost))
    cost, done = step(env, TL)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Turning Left Costs: {}".format(cost))
    cost, done = step(env, TR)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Turning Right Costs: {}".format(cost))
    cost, done = step(env, PK)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Picking Up Key Costs: {}".format(cost))
    cost, done = step(env, UD)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Unlocking Door Costs: {}".format(cost))

    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")

    # The number of steps so far
    print("Step Count: {}".format(env.step_count))



