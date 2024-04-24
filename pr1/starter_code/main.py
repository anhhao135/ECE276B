from utils import *
from example import example_use_of_gym_env
from dp_utils import *
from minigrid.core.world_object import Goal, Key, Door, Wall

import warnings
warnings.filterwarnings('ignore')

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


env_path = "./envs/known_envs/doorkey-5x5-normal.env"
env, info = load_env(env_path)  # load an environment

#cost, done = step(env, MF)
#cost, done = step(env, TL)
#cost, done = step(env, TL)
#cost, done = step(env, PK)
#cost, done = step(env, TR)

globalVisitedStates = np.array([[1,2,0,-1,0,0],[1,2,0,1,0,0], [1,3,0,-1,0,0]])
state = np.array([1,2,0,-1,0,0])
sum = globalVisitedStates - state
print(sum)
sum = np.sum(sum, axis=1)
print(sum)


currentState = getCurrentState(env,info)

nextPossibleStates = getNextPossibleStates(currentState,env)
print(nextPossibleStates)
filterOutUnpromisingNextPossibleStates(nextPossibleStates,1,env)






#print(getTypeInFront(env))
#print(getTypeLeft(env))
#print(getTypeRight(env))
#print(checkIfDeadEnd(env))
#print(getCurrentState(env, info))