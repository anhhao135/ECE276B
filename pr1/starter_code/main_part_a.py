from utils import *
from lib_part_a import *
import warnings
warnings.filterwarnings('ignore')
import os
from csv import writer


#this is to generate optimal policies for the known environments
#then use it to generate optimal sequences from start to finish
#save trajectories as gifs


env_dir = "envs/known_envs"

with open("known_envs_sequences.csv", "w+") as f_object: #save all optimal sequences in csv file
    writer_object = writer(f_object)
    for env_file in os.listdir(env_dir): #go through all the known environments
        env_path = os.path.join(env_dir,env_file)
        env, info = load_env(env_path)
        timeHorizon = 1000 #realistically the time horizon is equal to the number of possible states to guarantee termination, but this should happen well before that
        optimalPolicy = calculateOptimalPolicy(env, info, timeHorizon) #calculate the optimal policy for the environment and save it in a dictionary

        seq = [] #initialize empty optimal sequence to be generated per step

        t = 0 #keep track of the time step

        while not np.array_equal(info['goal_pos'], env.agent_pos): #keep on querying the optimal policy until the goal has been reached

            #UPDATE CURRENT STATE ----------------------------------------------------------------------------

            currentState = getCurrentState(env, info) #query the environment and construct the custom-defined state vector

            #UPDATE CURRENT STATE ----------------------------------------------------------------------------



            #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------

            optimalControl = optimalPolicy[np.array2string(currentState)[1:-1]] #the optimal policy will return the optimal control input

            seq.append(optimalControl) #add this control input to the sequence

            step(env, optimalControl) #perform this control input to the environment 

            t = t + 1 #increment the time step

            #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------

        env, info = load_env(env_path) #reset the environment state so the gif can draw it
        draw_gif_from_seq(seq, env, path = "gif_known/" + str(env_file) + ".gif")  # draw a GIF & save 
        writer_object.writerow([env_file, seq])
    f_object.close()
