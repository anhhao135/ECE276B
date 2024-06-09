from time import time
import numpy as np
import utils
import matplotlib.pyplot as plt
from my_utils import *
import scipy.sparse
from scipy.stats import multivariate_normal
from tqdm import tqdm
import ray


#construction of discrete state space
discreteStateSpace = np.array(constructDiscreteStateSpace())
discreteControlSpace = np.array(constructDiscreteControlSpace())


stateSpaceSize = discreteStateSpace.shape[0]
controlSpaceSize = discreteControlSpace.shape[0]

print("state space size:", stateSpaceSize)
print("control space size:", controlSpaceSize)
print("state space shape:", discreteStateSpace.shape)

L = np.zeros((stateSpaceSize,controlSpaceSize))
Q = 2 * scipy.sparse.eye(2)
R = 2 * scipy.sparse.eye(2)
q = 1

iterations = 1000

P_err = np.atleast_2d(discreteStateSpace[:,0:2].flatten())
Theta_err = discreteStateSpace[:,2]
U = np.atleast_2d(discreteControlSpace.flatten())


QBatch = scipy.sparse.kron(scipy.sparse.eye(stateSpaceSize),Q)
P_err_diag = scipy.sparse.diags(P_err[0], 0)
L_P_err = P_err @ QBatch @ P_err_diag
L_P_err = L_P_err.reshape((-1, 2))
L_P_err = np.atleast_2d(np.sum(L_P_err, axis=1)).T

L_Theta_err = np.atleast_2d(q * np.square(np.ones(stateSpaceSize) - np.cos(Theta_err))).T

RBatch = scipy.sparse.kron(scipy.sparse.eye(controlSpaceSize),R)
U_diag = scipy.sparse.diags(U[0], 0)
L_U = U @ RBatch @ U_diag
L_U = L_U.reshape((-1, 2))
L_U = np.atleast_2d(np.sum(L_U, axis=1)).T


L = np.tile(L_P_err, controlSpaceSize) + np.tile(L_Theta_err, controlSpaceSize) + np.tile(L_U, stateSpaceSize).T




mu = np.zeros(3)
print(multivariate_normal.pdf(np.array([0.2,0.2,0.01]), mu, utils.sigma))

traj = utils.lissajous

referenceStatesAhead = []

for i in range(0, 101):
    referenceStatesAhead.append(traj(i))


#matchState = np.array([0.01,0.02,0.03,10])
#indexes = getNeighboringStates(matchState, discreteStateSpace, 8)
#print(discreteStateSpace[indexes])

@ray.remote
def f(i):
    nextStates = []
    for j in range(controlSpaceSize):
        currentState = discreteStateSpace[i]
        currentTime = int(currentState[3])
        currentControl = discreteControlSpace[j]
        nextState = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead[currentTime], referenceStatesAhead[currentTime+1])    
        nextStates.append(nextState)
    return nextStates

def g(i):
    nextStates = []
    for j in range(controlSpaceSize):
        currentState = discreteStateSpace[i]
        currentTime = int(currentState[3])
        currentControl = discreteControlSpace[j]
        nextState = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead[currentTime], referenceStatesAhead[currentTime+1])    
        nextStates.append(nextState)
    return nextStates






startTime = time()

results = []

for i in range(stateSpaceSize):
    results.append(f.remote(i))

ray.get(results)

rayFinishTime = time()

for i in range(stateSpaceSize):
    print(i)
    g(i)

nativeFinishTime = time()


print("ray time", rayFinishTime - startTime)
print("native time", nativeFinishTime - rayFinishTime)

#P = scipy.sparse.zeros((stateSpaceSize,controlSpaceSize,stateSpaceSize)) #initialize motion model matrix: P[state t,control input, state t+1] = probability of occurence
Q = scipy.sparse.csr_matrix(L)



while False:

    L = np.zeros((25,4)) #initialize all stage costs to 0
    #now populate the stage cost

    #first do out of map costs
    L[0:5,0] = 1 #top row move north would bring it out of map, incurs cost of 1
    L[[4,9,14,19,24],1] = 1 #rightmost column move east would bring it out of map, incurs cost of 1
    L[20:25,2] = 1 #bottom row move south would bring it out of map, incurs cost of 1
    L[[0,5,10,15,20],3] = 1 #leftmost column move west would bring it out of map, incurs cost of 1

    #second do special costs
    L[1,:] = -10 #special state A, all controls incur -10
    L[3,:] = -5 #special state B, all controls incur -5


    P = np.zeros((stateSpaceSize,controlSpaceSize,stateSpaceSize)) #initialize motion model matrix: P[state t,control input, state t+1] = probability of occurence
    V = np.zeros((iterations+1, stateSpaceSize))
    pi = np.zeros((iterations+1,stateSpaceSize),dtype='int')
    Q = np.zeros((iterations+1, stateSpaceSize, controlSpaceSize))