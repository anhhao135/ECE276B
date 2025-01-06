from utils import *
from lib_part_a import *
import warnings
warnings.filterwarnings('ignore')
import os
from create_env import *

#this is to test the perfomance of the policy in part A by customizing the agent's initial orientation for the known maps

map_name = "doorkey-8x8-shortcut" #pick the known map

#choose the agent initial pose
agent_start_pos = (1,5)
agent_start_dir = UP

create_performance_envs(map_name, agent_start_pos, agent_start_dir) #generate the environment


env_dir = "envs/performance_envs"
env_file = map_name + ".env"

env_path = os.path.join(env_dir,env_file)
env, info = load_env(env_path)
timeHorizon = 1000
optimalPolicy = calculateOptimalPolicy(env, info, timeHorizon) #calculate the policy

seq = [] #initialize empty optimal sequence to be generated per step

t = 0 #keep track of the time step

while not np.array_equal(info['goal_pos'], env.agent_pos): #keep on querying the optimal policy until the goal has been reached

    #UPDATE CURRENT STATE ----------------------------------------------------------------------------

    currentState = getCurrentState(env, info)

    #UPDATE CURRENT STATE ----------------------------------------------------------------------------


    #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------

    optimalControl = optimalPolicy[np.array2string(currentState)[1:-1]] #the optimal policy will return the optimal control input

    seq.append(optimalControl) #add this control input to the sequence

    step(env, optimalControl) #perform this control input to the environment 

    t = t + 1 #increment the time step

    #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------


env, info = load_env(env_path) #reset the environment state so the gif can draw it
draw_gif_from_seq(seq, env, path = "gif_performance/" + str(env_file) + ".gif")  # draw a GIF & save 

