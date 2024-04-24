from utils import *
from example import example_use_of_gym_env
from dp_utils import *
from minigrid.core.world_object import Goal, Key, Door, Wall
import time

import warnings
warnings.filterwarnings('ignore')

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


#env_path = "./envs/known_envs/doorkey-5x5-normal.env"
env_path = "./envs/known_envs/doorkey-6x6-normal.env"
env, info = load_env(env_path)  # load an environment

currentStates = np.atleast_2d(getCurrentState(env,info))
visitedStates = currentStates.copy()
print("initial visited states")
print(visitedStates)
timeHorizon = 1000


for t in range(timeHorizon):
    print("------------------------")
    print(t)
    nextStates = []
    for currentState in currentStates:
        nextPossibleStates = getNextPossibleStates(currentState, env)
        for nextPossibleState in nextPossibleStates:
            if not (np.array_equal(nextPossibleState, np.zeros(6, dtype=np.int16))):
                if not checkIfStateBeenVisited(nextPossibleState, visitedStates):
                    print("next possible state")
                    print(nextPossibleState)
                    visitedStates = np.vstack((visitedStates, nextPossibleState))
                    nextStates.append(nextPossibleState)
    currentStates = np.array(nextStates)
    print("current states")
    print(currentStates)
    print("visited states")
    print(visitedStates)
    print("------------------------")