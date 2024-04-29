from lib_part_b import *
from utils import *
import warnings
warnings.filterwarnings('ignore')
import os
from csv import writer
from natsort import natsorted

env_dir = "envs/random_envs"
#this is to run the calculation for part B on all random environments


#define the parameters for generating the random maps
#--------------------------------------------------------------
goalLocations = [np.array([5,1]), np.array([6,3]), np.array([5,6])] #these can occur with equal probabilities
keyLocations = [np.array([1,1]), np.array([2,3]), np.array([1,6])] #these can occur with equal probabilities
doorLocations = [np.array([4,2]), np.array([4,5])] #this is fixed

initialPos = np.array([3,5], dtype=np.int16) #this is fixed
initialDir = np.array([0,-1], dtype=np.int16) #this is fixed
timeHorizon = 1000 #the policy calculation should terminate earlier than the horizon
dimension = 8
#--------------------------------------------------------------

#calculate the optimal policy once based on the random map generation parameters

optimalPolicy = calculateSingleOptimalPolicy(timeHorizon, goalLocations, keyLocations, doorLocations, dimension, initialPos, initialDir)

with open("random_envs_sequences.csv", "w+") as f_object: #save all optimal sequences in csv file
    writer_object = writer(f_object)
    for env_file in natsorted(os.listdir(env_dir)):
        env_path = os.path.join(env_dir,env_file)
        env, info = load_env(env_path)

        #once the random environment has been loaded, get the initial state

        door1 = env.grid.get(4,2)
        door2 = env.grid.get(4,5)
        doorState = np.array([door1.is_open, door2.is_open], dtype=np.int16) #construct the doors' lock state
        keyPos = info['key_pos'] + np.array([1,1]) #get the key location added to [1,1] because of the surrounding wall padding; this is just how the custom state vector is defined
        goalPos = info['goal_pos'] + np.array([1,1]) #get the goal location added to [1,1]

        #initialized the door unlock attempts to both 0
        door1StateAfter = door1.is_open
        door2StateAfter = door2.is_open
        door1UnlockAttempted = 0
        door2UnlockAttempted = 0

        seq = [] #initialize empty optimal sequence to be generated per step

        t = 0 #keep track of the time step

        while not np.array_equal(info['goal_pos'], env.agent_pos): #keep on querying the optimal policy until the goal has been reached

            #UPDATE CURRENT STATE ----------------------------------------------------------------------------

            #this logic is to detect whenever a door unlock has been attempted, and which one, to change the state vector accordingly
            #--------------
            door1UnlockAttemptedRisingEdge = door1.is_open != door1StateAfter
            door2UnlockAttemptedRisingEdge = door2.is_open != door2StateAfter

            if (door1UnlockAttemptedRisingEdge or door2UnlockAttemptedRisingEdge): #this is to latch the unlock attempt on the pertinent door
                door1UnlockAttempted = door1.is_open != door1StateAfter
                door2UnlockAttempted = door2.is_open != door2StateAfter

            door1StateAfter = door1.is_open
            door2StateAfter = door2.is_open
            #--------------
            

            currentPos = env.agent_pos + np.array([1,1]) #update the current agent position state vector portion
            currentDir = env.dir_vec #update the current agent facing direction state vector portion
            currentKeyPickedUp = env.carrying is not None #update if the agent has picked up the key state vector portion
            currentState = createCurrentStateVector(currentPos, currentDir, goalPos, keyPos, doorState, currentKeyPickedUp, door1UnlockAttempted, door2UnlockAttempted) #construct the custom current state vector based on the current environment at this step

            lookupState = np.concatenate((np.array([t],dtype=np.int16), currentState)) #construct the key to look up what control input to use next in the optimal policy dictionary; this is the current state and time
            lookupState = np.array2string(lookupState)[1:-1]

            #UPDATE CURRENT STATE ----------------------------------------------------------------------------

            

            #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------

            optimalControl = optimalPolicy[lookupState] #the optimal policy will return the optimal control input

            seq.append(optimalControl) #add this control input to the sequence

            step(env, optimalControl) #perform this control input to the environment 

            t = t + 1 #increment the time step

            #GET CONTROL INPUT FROM POLICY -------------------------------------------------------------------

        env, info = load_env(env_path) #reset the environment state so the gif can draw it
        draw_gif_from_seq(seq, env, path = "gif_random/" + str(env_file) + ".gif")  # draw a GIF & save 
        writer_object.writerow([env_file, seq])
    f_object.close()





