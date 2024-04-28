from dp_utils_part_b import *
from utils import *
from example import example_use_of_gym_env
from minigrid.core.world_object import Goal, Key, Door, Wall
import time
import random
import warnings
warnings.filterwarnings('ignore')


goalLocations = [np.array([5,1]), np.array([6,3]), np.array([5,6])]
keyLocations = [np.array([1,1]), np.array([2,3]), np.array([1,6])]
doorLocations = [np.array([4,2]), np.array([4,5])]
dimension = 8
randomMap = constructRandomMap(goalLocations, keyLocations, doorLocations, dimension)
print(randomMap.T)

goalLocations = np.array(goalLocations, dtype=np.int16) + np.array([1,1])
keyLocations = np.array(keyLocations, dtype=np.int16) + np.array([1,1])
doorStates = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.int16)
initialPos = np.array([3,5], dtype=np.int16) + np.array([1,1])
initialDir = np.array([0,-1], dtype=np.int16)

initialStates =  []

for goalLocation in goalLocations:
    for keyLocation in keyLocations:
        for doorState in doorStates:
            initialState = createStateVector(initialPos, initialDir, goalLocation, keyLocation, doorState, 0)
            initialStates.append(initialState)

optimalPolicy = calculateSingleOptimalPolicy(randomMap, initialStates, 20)
print(len(optimalPolicy))


env, info = load_env("./envs/random_envs/doorkey-8x8-4.env")
print(info)

# Visualize the environment
#plot_env(env)

door1 = env.grid.get(4,2)
door2 = env.grid.get(4,5)
doorState = np.array([door1.is_open, door2.is_open], dtype=np.int16)
keyPos = info['key_pos'] + np.array([1,1])
goalPos = info['goal_pos'] + np.array([1,1])

timeHorizon = 20

door1StateAfter = door1.is_open
door2StateAfter = door2.is_open
door1UnlockAttempted = 0
door2UnlockAttempted = 0

for t in range(timeHorizon):

    door1UnlockAttemptedRisingEdge = door1.is_open != door1StateAfter
    door2UnlockAttemptedRisingEdge = door2.is_open != door2StateAfter

    if (door1UnlockAttemptedRisingEdge or door2UnlockAttemptedRisingEdge):
        door1UnlockAttempted = door1.is_open != door1StateAfter
        door2UnlockAttempted = door2.is_open != door2StateAfter

    door1StateAfter = door1.is_open
    door2StateAfter = door2.is_open
    currentPos = env.agent_pos + np.array([1,1])
    currentDir = env.dir_vec
    currentKeyPickedUp = env.carrying is not None
    currentState = createCurrentStateVector(currentPos, currentDir, goalPos, keyPos, doorState, currentKeyPickedUp, door1UnlockAttempted, door2UnlockAttempted)
    print(currentState)
    lookupState = np.concatenate((np.array([t],dtype=np.int16), currentState))
    lookupState = np.array2string(lookupState)[1:-1]
    optimalControl = optimalPolicy[lookupState]
    print(optimalControl)
    cost, done = step(env, optimalControl)  # MF=0, TL=1, TR=2, PK=3, UD=4
    plot_env(env)
    
