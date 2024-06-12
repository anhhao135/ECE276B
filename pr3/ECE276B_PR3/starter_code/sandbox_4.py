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
import multiprocessing



#construction of discrete state space
discreteStateSpace = np.array(constructDiscreteStateSpace())
discreteControlSpace = np.array(constructDiscreteControlSpace())


stateSpaceSize = discreteStateSpace.shape[0]
perTimeStateSpaceSize = int(stateSpaceSize / utils.T)
controlSpaceSize = discreteControlSpace.shape[0]

numberOfNeighbors = 2

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


numberOfCPUs = multiprocessing.cpu_count()
print("number of cpus", numberOfCPUs)
stateSize = stateSpaceSize
jobStateSpaces = []
leftOverJobs = stateSize % numberOfCPUs
statesPerJob = stateSize // numberOfCPUs
if leftOverJobs != 0:
    statesPerJob = statesPerJob + 1
leftOverJobs_ = stateSpaceSize - (numberOfCPUs - 1) * statesPerJob

for i in range(numberOfCPUs):
    if i == numberOfCPUs - 1 and leftOverJobs != 0:
        jobStateSpaces.append(np.arange(i * statesPerJob, i * statesPerJob + leftOverJobs_, 1))
    else:
        jobStateSpaces.append(np.arange(i * statesPerJob, i * statesPerJob + statesPerJob, 1))

print(jobStateSpaces)

for i in jobStateSpaces[-1]:
    print(i)

startTime = time()

@ray.remote
def job(stateSpaceArray):
    P_job = np.zeros((stateSpaceArray.shape[0],controlSpaceSize,numberOfNeighbors+1))
    for i in stateSpaceArray:
        for j in range(controlSpaceSize):
            currentState = discreteStateSpace[i]
            currentControl = discreteControlSpace[j]
            currentTime = int(currentState[3])
            nextTime = (currentTime + 1) % T
            referenceStatesAhead_ = referenceStatesAhead[currentTime:currentTime + 10]
            referenceStatesAheadThetas = referenceStatesAhead_[:,2]
            referenceStatesAheadThetas = np.concatenate((referenceStatesAheadThetas, np.atleast_1d(currentState[2])))
            referenceStatesAheadThetas = np.unwrap(referenceStatesAheadThetas)
            referenceStatesAhead_[:,2] = referenceStatesAheadThetas[:-1]
            currentState[2] = referenceStatesAheadThetas[-1]
            nextStateContinuous = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead_[0], referenceStatesAhead_[1], True)
            nextStateDiscrete = continuousToDiscreteState(np.array(nextStateContinuous).T, discreteStateSpace[:perTimeStateSpaceSize,:3])
            nextStateNeighborsProbVectorFull = stateNeighbors[nextStateDiscrete]
            nextStateNeighborsProbVectorNonZeroIndexes = np.nonzero(nextStateNeighborsProbVectorFull > 0)
            nextStateNeighborsProbVectorNonZero = nextStateNeighborsProbVectorFull[nextStateNeighborsProbVectorNonZeroIndexes]
            V_mask[i, j, :] = np.array(nextStateNeighborsProbVectorNonZeroIndexes) + nextTime * perTimeStateSpaceSize
            P_job[i - stateSpaceArray[0],j,:] = nextStateNeighborsProbVectorNonZero

    #print("done with", stateSpaceArray[0])
    return P_job


ray.init()
jobs = [job.remote(stateSpace) for stateSpace in jobStateSpaces]
results = ray.get(jobs)

for i in range(numberOfCPUs):
    statesInJob = jobStateSpaces[i]
    print(statesInJob)
    P[statesInJob[0]:statesInJob[-1]+1,:,:] = results[i]

rayEndTime = time()
print("ray time", rayEndTime - startTime)
statesInJob = jobStateSpaces[15]

print("ray")

print(P[198,:,:])
print(P[199,:,:])
print(P[200,:,:])
print(P[201,:,:])



P_ray = P.copy()

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
        
        nextStateContinuous = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead_[0], referenceStatesAhead_[1], True)

        nextStateDiscrete = continuousToDiscreteState(np.array(nextStateContinuous).T, discreteStateSpace[:perTimeStateSpaceSize,:3])

        nextStateNeighborsProbVectorFull = stateNeighbors[nextStateDiscrete]
        nextStateNeighborsProbVectorNonZeroIndexes = np.nonzero(nextStateNeighborsProbVectorFull > 0)
        nextStateNeighborsProbVectorNonZero = nextStateNeighborsProbVectorFull[nextStateNeighborsProbVectorNonZeroIndexes]

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


print("native")
print(P[198,:,:])
print(P[199,:,:])
print(P[200,:,:])
print(P[201,:,:])


while False:

    nativeEndTime = time()
    print("native time", nativeEndTime - rayEndTime)

    print("array equal", np.array_equal(P, P_ray))

    statesInJob = jobStateSpaces[15]
    print(P[statesInJob[-1],-1,:])

while False:

    for i in range(numberOfCPUs):
        statesInJob = jobStateSpaces[i]
        print(i)
        print("P",P[statesInJob[0]:statesInJob[-1]+1,:,:].shape)
        print("ray",results[i].shape)
        print("array equal", np.array_equal(P[statesInJob[0]:statesInJob[-1]+1,:,:], results[i]))