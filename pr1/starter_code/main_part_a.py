from utils import *
from lib_part_a import *
import warnings
warnings.filterwarnings('ignore')
import os

env_dir = "envs/known_envs"
#this is to run the calculation for part A on all the known environments

for env_file in os.listdir(env_dir):
    env_path = os.path.join(env_dir,env_file)
    print(env_path)
    env, info = load_env(env_path)  # load an environment
    timeHorizon = 400 #the time horizon for a 5x5 environment, this is the worst case since if the agent has managed to go through all the nodes, then it must have reached the goal
    #the time horizon maximum is (map dimension)^2 * 4 * 2 * 2, the number of possible states
    #but we expect the algorithm to terminate well before this at the value function, which is the optimal cost-to-go from start to goal
    seq = calculateOptimalSequence(env, info, timeHorizon)
    print(seq)
    draw_gif_from_seq(seq, env, path = "gif/" + str(env_file) + ".gif")  # draw a GIF & save 