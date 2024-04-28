from lib_part_b import *
from utils import *
import warnings
warnings.filterwarnings('ignore')
import os
from csv import writer

env_dir = "envs/random_envs"
#this is to run the calculation for part B on all random environments

goalLocations = [np.array([5,1]), np.array([6,3]), np.array([5,6])]
keyLocations = [np.array([1,1]), np.array([2,3]), np.array([1,6])]
doorLocations = [np.array([4,2]), np.array([4,5])]
initialPos = np.array([3,5], dtype=np.int16)
initialDir = np.array([0,-1], dtype=np.int16)
timeHorizon = 30
dimension = 8

#calculate the optimal policy once based on the random map generation parameters
optimalPolicy = calculateSingleOptimalPolicy(timeHorizon, goalLocations, keyLocations, doorLocations, dimension, initialPos, initialDir)

with open('random_envs_sequences.csv', 'a') as f_object: #save all optimal sequences in csv file
    writer_object = writer(f_object)
    for env_file in os.listdir(env_dir):
        env_path = os.path.join(env_dir,env_file)
        env, info = load_env(env_path)
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

        seq = []

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
            lookupState = np.concatenate((np.array([t],dtype=np.int16), currentState))
            lookupState = np.array2string(lookupState)[1:-1]
            optimalControl = optimalPolicy[lookupState]
            seq.append(optimalControl)
            cost, done = step(env, optimalControl)  # MF=0, TL=1, TR=2, PK=3, UD=4
            if np.array_equal(info['goal_pos'], env.agent_pos):
                print("GOAL REACHED")
                break

        print(seq)
        env, info = load_env(env_path)
        draw_gif_from_seq(seq, env, path = "gif_random/" + str(env_file) + ".gif")  # draw a GIF & save 
        writer_object.writerow([env_file, seq])
    f_object.close()





