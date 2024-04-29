from utils import *
from lib_part_a import *
import warnings
warnings.filterwarnings('ignore')
import os
from csv import writer

env, info = load_env("envs/known_envs/doorkey-6x6-shortcut.env")  # load an environment
timeHorizon = 20 #the time horizon for a 5x5 environment, this is the worst case since if the agent has managed to go through all 
#possiblePositions = [np.array([1,2],dtype=np.int16), np.array([1,3],dtype=np.int16), np.array([3,1],dtype=np.int16), np.array([3,2],dtype=np.int16)]
possiblePositions = [np.array([2,3],dtype=np.int16)]
policy = calculateOptimalSequence(env, info, timeHorizon, possiblePositions)
print(policy)

door = env.grid.get(info["door_pos"][0], info["door_pos"][1])

seq = []

print(info['goal_pos'])



while not np.array_equal(info['goal_pos'], env.agent_pos): #keep on querying the optimal policy until the goal has been reached

    #UPDATE CURRENT STATE ----------------------------------------------------------------------------

    currentPos = env.agent_pos
    currentDir = env.dir_vec
    currentKeyPickedUp = env.carrying is not None #update if the agent has picked up the key state vector portion
    doorState = door.is_open
    currentState = createStateVector(currentPos, currentDir, doorState, currentKeyPickedUp)#construct the custom current state vector based on the current environment at this step
    lookupState = np.array2string(currentState)[1:-1]

    #UPDATE CURRENT STATE ----------------------------------------------------------------------------

    #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------

    optimalControl = policy[lookupState][0] #the optimal policy will return the optimal control input

    seq.append(optimalControl) #add this control input to the sequence

    step(env, optimalControl) #perform this control input to the environment 
    print(optimalControl)

    #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------