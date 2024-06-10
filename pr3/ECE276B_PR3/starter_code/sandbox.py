from time import time
import numpy as np
import utils
import matplotlib.pyplot as plt
from my_utils import *
import scipy.sparse
from scipy.stats import multivariate_normal
from tqdm import tqdm
import ray
import sparse
import sys

print("here")

#construction of discrete state space
discreteStateSpace = np.array(constructDiscreteStateSpace())
discreteControlSpace = np.array(constructDiscreteControlSpace())


stateSpaceSize = discreteStateSpace.shape[0]
perTimeStateSpaceSize = int(stateSpaceSize / utils.T)
controlSpaceSize = discreteControlSpace.shape[0]

numberOfNeighbors = 8

stateNeighbors = findNeighborsOfState(discreteStateSpace, perTimeStateSpaceSize, numberOfNeighbors)

#print(discreteStateSpace[5500])
#print(stateNeighbors[5500 % perTimeStateSpaceSize])
#print(2*perTimeStateSpaceSize + np.array(np.nonzero(stateNeighbors[5500 % perTimeStateSpaceSize] > 0)))

#print(stateNeighbors[5500])
#print(np.nonzero(stateNeighbors[5500] > 0))
#print(stateNeighbors[5500].shape)

print("state space size:", stateSpaceSize)
print("control space size:", controlSpaceSize)
print("per time state space shape:", perTimeStateSpaceSize)

L = np.zeros((stateSpaceSize,controlSpaceSize))
Q = 2 * scipy.sparse.eye(2)
R = 1 * scipy.sparse.eye(2)
q = 1


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

#mu = np.zeros(3)
#print(multivariate_normal.pdf(np.array([0.2,0.2,0.01]), mu, utils.sigma))

traj = utils.lissajous

referenceStatesAhead = []

for i in range(0, 120):
    referenceStatesAhead.append(traj(i))

referenceStatesAhead = np.array(referenceStatesAhead)


P = np.zeros((stateSpaceSize,controlSpaceSize,numberOfNeighbors+1))
V_mask = np.zeros((stateSpaceSize,controlSpaceSize,numberOfNeighbors+1), dtype=np.uint16)


for i in range(stateSpaceSize):
    for j in range(controlSpaceSize):
        #print("--------------")
        currentState = discreteStateSpace[i]
        currentControl = discreteControlSpace[j]
        currentTime = int(currentState[3])
        nextTime = (currentTime + 1) % T
        referenceStatesAhead_ = referenceStatesAhead[currentTime:currentTime + 10]
        referenceStatesAheadThetas = referenceStatesAhead_[:,2]
        #print("current state", currentState)
        referenceStatesAheadThetas = np.concatenate((referenceStatesAheadThetas, np.atleast_1d(currentState[2])))
        #print(referenceStatesAheadThetas)
        referenceStatesAheadThetas = np.unwrap(referenceStatesAheadThetas)
        #print(referenceStatesAheadThetas)
        referenceStatesAhead_[:,2] = referenceStatesAheadThetas[:-1]
        currentState[2] = referenceStatesAheadThetas[-1]
        
        nextStateContinuous = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead_[0], referenceStatesAhead_[1], False)


        nextStateDiscrete = continuousToDiscreteState(np.array(nextStateContinuous).T, discreteStateSpace[:perTimeStateSpaceSize,:3])

        nextStateNeighborsProbVectorFull = stateNeighbors[nextStateDiscrete]
        nextStateNeighborsProbVectorNonZeroIndexes = np.nonzero(nextStateNeighborsProbVectorFull > 0)
        nextStateNeighborsProbVectorNonZero = nextStateNeighborsProbVectorFull[nextStateNeighborsProbVectorNonZeroIndexes]

        if (np.array_equal(currentState, np.array([0,0,0,0]))):
            print("current state",currentState)
            print("current control", currentControl)
            print("next state", nextStateContinuous)
            print("next state discrete", discreteStateSpace[nextStateDiscrete + perTimeStateSpaceSize])
            print("next state discrete index", nextStateDiscrete + perTimeStateSpaceSize)
            print(np.array(nextStateNeighborsProbVectorNonZeroIndexes) + perTimeStateSpaceSize)
            print(nextStateNeighborsProbVectorNonZero)
            print("\n\n")

        V_mask[i, j, :] = np.array(nextStateNeighborsProbVectorNonZeroIndexes) + nextTime * perTimeStateSpaceSize

        #print("next state discrete neighbor non zero likelihoods", nextStateNeighborsProbVectorNonZero)
        P[i,j,:] = nextStateNeighborsProbVectorNonZero
        #print("next state discrete", nextStateDiscrete)
        #print("next state discrete neighbor likelihoods", nextStateNeighborsProbVectorFull)
        #print(discreteStateSpace[5500])
        #print(stateNeighbors[5500 % perTimeStateSpaceSize])
        #print(2*perTimeStateSpaceSize + np.array(np.nonzero(stateNeighbors[5500 % perTimeStateSpaceSize] > 0)))
        likelihoodVector = stateNeighbors[nextStateDiscrete % perTimeStateSpaceSize]
        #P[i,j,:] = stateNeighstateNeighbors[nextStateDiscrete % perTimeStateSpaceSize]
        #print("next state continuous", np.vstack((nextStateContinuous, nextTime)).T)
        #print("next states discrete", discreteStateSpace[nextStatesIndexesDiscrete])
        #print("--------------")
    print(i)


np.save('P.npy', P)
np.save('V_mask.npy', V_mask)
iterations = 200

V = np.zeros((iterations+1, stateSpaceSize))
pi = np.zeros((iterations+1,stateSpaceSize),dtype=np.uint16)

gamma = 0.95


for k in range(iterations):
    Q = np.zeros((stateSpaceSize,controlSpaceSize))
    V_k = V[k,:][V_mask]
    Q = L + gamma * np.sum(np.multiply(P,V_k), axis=2)
    pi[k+1,:] = np.argmin(Q, axis=1) #policy improvement
    V[k+1,:] = np.min(Q, axis=1) #value update
    maxValueDifference = np.abs(V[k+1] - V[k]).max()

    discreteState = continuousToDiscreteState(np.array([0,0,0,0]),discreteStateSpace)
    print("control for no error at time 0", discreteControlSpace[int(pi[k+1,discreteState])])
    print("Q for discrete state", Q[discreteState])


    print("max value difference", maxValueDifference)
    print("iteration", k)

np.savetxt('policy.txt', pi[-1:])
np.savetxt('controlSpace.txt', discreteControlSpace)
np.savetxt('stateSpace.txt', discreteStateSpace)



while False:
    Q = np.zeros((stateSpaceSize,controlSpaceSize))

    for i in range(stateSpaceSize):
        for j in range(controlSpaceSize):
            likelihoods = P[i,j,:]
            #nextStateNeighborsProbVectorFull = stateNeighbors[motionModel[]]
            nextStateDiscrete = np.nonzero(nextStateNeighborsProbVectorFull > 0)
            print(i % perTimeStateSpaceSize)
            print(likelihoods)
            print(nextStateDiscrete)


#pi[k+1,:] = np.argmin(Q, axis=1) #policy improvement

while False:
        
    for i in range(stateSpaceSize):
        for j in range(controlSpaceSize):
            #print("--------------")
            currentState = discreteStateSpace[i]
            currentControl = discreteControlSpace[j]
            currentTime = int(currentState[3])
            nextTime = (currentTime + 1) % 100
            nextStateContinuous = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead[currentTime], referenceStatesAhead[currentTime+1])
            nextStateDiscrete = continuousToDiscreteState(nextStateContinuous, discreteControlSpace)
            P[i,j,:] = stateNeighbors[nextStateDiscrete]
            #print("next state continuous", np.vstack((nextStateContinuous, nextTime)).T)
            #print("next states discrete", discreteStateSpace[nextStatesIndexesDiscrete])
            #print("--------------")
        print(i)


while False:
    for i in range(stateSpaceSize):

        for j in range(controlSpaceSize):
            print("--------------")
            currentState = discreteStateSpace[i]
            currentTime = int(currentState[3])
            nextTime = (currentTime + 1) % 100
            currentControl = discreteControlSpace[j]
            nextStateContinuous = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead[currentTime], referenceStatesAhead[currentTime+1])
            nextStatesIndexesDiscrete = continuousToDiscreteStateAndNeighbors(np.vstack((nextStateContinuous, nextTime)).T, discreteStateSpace, numberOfNeighbors)  
            print("next state continuous", np.vstack((nextStateContinuous, nextTime)).T)
            print("next states discrete", discreteStateSpace[nextStatesIndexesDiscrete])
            print("--------------")
            



while False:
    for i in range(1000):
        matchState = np.array([0.01,0.02,0.03,10])
        indexes = getNeighboringStates(matchState, discreteStateSpace, 4)
        print(i)


while False:

    nextStates = []
    



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