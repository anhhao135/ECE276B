from utils import *
from example import example_use_of_gym_env
from dp_utils import *
from minigrid.core.world_object import Goal, Key, Door, Wall
import time
import random

import warnings
warnings.filterwarnings('ignore')

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


#env_path = "./envs/known_envs/doorkey-5x5-normal.env"
#env_path = "./envs/known_envs/doorkey-5x5-normal.env"
#env, info = load_env(env_path)  # load an environment


env = generate_random_env(random.randint(0,1000), 'MiniGrid-DoorKey-8x8-v0')
env, info = load_env_generated(env)  # load an environment

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

#draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save
draw_gif_from_seq(seq, env)  # draw a GIF & save