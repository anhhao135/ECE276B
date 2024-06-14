from time import time
import numpy as np
import utils
from my_utils import *
import ray
import multiprocessing

#script to precalculate the transition matrix required for GPI
#uses Ray to parallelize process

startTime = time()

#construction of discrete state space and control space
#specify discretization count and shrink factors (discussed in report)
discreteStateSpace = np.array(constructDiscreteStateSpace(sparseDiscretizationCount = 5, densePositionErrorShrinkFactor = 0.3, denseThetaErrorShrinkFactor = 0.3))
discreteControlSpace = np.array(constructDiscreteControlSpace(sparseControlDiscretizationCount = 5, densesControlVShrinkFactor = 0.3, denseControlWShrinkFactor = 0.3))
stateSpaceSize = discreteStateSpace.shape[0]
perTimeStateSpaceSize = int(stateSpaceSize / utils.T)
controlSpaceSize = discreteControlSpace.shape[0]

#specify neighbor count
numberOfNeighbors = 8

#precompute the likelihoods of the neighbors surround each state given the state is the mean
stateNeighbors = findNeighborsOfState(discreteStateSpace, perTimeStateSpaceSize, numberOfNeighbors)

print("State space size:", stateSpaceSize)
print("Control space size:", controlSpaceSize)
print("Per-time state space size:", perTimeStateSpaceSize) #how many states within each time step?

traj = utils.lissajous

#precompute reference trajectory for all time steps + +1
referenceStatesAhead = []
for i in range(0, 101):
    referenceStatesAhead.append(traj(i))
referenceStatesAhead = np.array(referenceStatesAhead)   

#initialize empty transition and mask matrix
P = np.zeros((stateSpaceSize,controlSpaceSize,numberOfNeighbors+1))
V_mask = np.zeros((stateSpaceSize,controlSpaceSize,numberOfNeighbors+1), dtype=np.uint16)


#logic to divide calculation of P into equal worker chunks split between available cpu cores
numberOfCPUs = multiprocessing.cpu_count()
stateSize = stateSpaceSize
jobStateSpaces = []
leftOverJobs = stateSize % numberOfCPUs
statesPerJob = stateSize // numberOfCPUs
if leftOverJobs != 0:
    statesPerJob = statesPerJob + 1
leftOverJobs_ = stateSpaceSize - (numberOfCPUs - 1) * statesPerJob

#each worker will be responsible for a range of states in the total state space
for i in range(numberOfCPUs):
    if i == numberOfCPUs - 1 and leftOverJobs != 0:
        jobStateSpaces.append(np.arange(i * statesPerJob, i * statesPerJob + leftOverJobs_, 1))
    else:
        jobStateSpaces.append(np.arange(i * statesPerJob, i * statesPerJob + statesPerJob, 1))


#define ray worker function
@ray.remote
def job(stateSpaceArray):
        P_job = np.zeros((stateSpaceArray.shape[0],controlSpaceSize,numberOfNeighbors+1))
        #initialize partial transition matrix to be populated with values
        V_mask_job = np.zeros((stateSpaceArray.shape[0],controlSpaceSize,numberOfNeighbors+1), dtype=np.uint16)
        #initialize partial mask matrix to be populated with values
        #iterate through all possible state-control pairs
        for i in stateSpaceArray:
            for j in range(controlSpaceSize):

                #get current state, control, and time
                currentState = discreteStateSpace[i].copy()
                currentControl = discreteControlSpace[j]
                currentTime = int(currentState[3])
                nextTime = (currentTime + 1) % T

                #logic to ensure rotational error is on a continuous flow instead of being wrapped around at -pi or pi
                referenceStatesAhead_ = referenceStatesAhead[currentTime:currentTime + 2].copy()
                referenceStatesAheadThetas = referenceStatesAhead_[:,2]
                referenceStatesAheadThetas = np.concatenate((referenceStatesAheadThetas, np.atleast_1d(currentState[2])))
                referenceStatesAheadThetas = np.unwrap(referenceStatesAheadThetas)
                referenceStatesAhead_[:,2] = referenceStatesAheadThetas[:-1]
                currentState[2] = referenceStatesAheadThetas[-1]

                #get next state from error motion model
                nextStateContinuous = errorMotionModelNoNoise(utils.time_step, currentState[0:2], currentState[2], currentControl, referenceStatesAhead_[0], referenceStatesAhead_[1], False)
                #convert this continuous state into a discrete index
                nextStateDiscrete = continuousToDiscreteState(np.array(nextStateContinuous).T, discreteStateSpace[:,:3])
                #get the likelihoods of the next state and its neighbors, normalized
                nextStateNeighborsProbVectorFull = stateNeighbors[nextStateDiscrete]
                #only use the non-zero probabilities
                nextStateNeighborsProbVectorNonZeroIndexes = np.nonzero(nextStateNeighborsProbVectorFull > 0)
                nextStateNeighborsProbVectorNonZero = nextStateNeighborsProbVectorFull[nextStateNeighborsProbVectorNonZeroIndexes]
                #mask the value function so that it only corresponds to the next state and its valid neighbors
                V_mask_job[i - stateSpaceArray[0], j, :] = np.array(nextStateNeighborsProbVectorNonZeroIndexes) + nextTime * perTimeStateSpaceSize
                #populate the partial transition matrix with the likelihoods of the next vector and its neighbors
                P_job[i - stateSpaceArray[0],j,:] = nextStateNeighborsProbVectorNonZero

        return (P_job, V_mask_job)



#create parallel ray workers
ray.init()
jobs = [job.remote(stateSpace) for stateSpace in jobStateSpaces] #pass in the states each worker is responsible for
results = ray.get(jobs)
#wait until all workers done
for i in range(numberOfCPUs):
    #get the partial transition and mask matrices of each worker and place them into the master matrices
    statesInJob = jobStateSpaces[i]
    P[statesInJob[0]:statesInJob[-1]+1,:,:] = results[i][0]
    V_mask[statesInJob[0]:statesInJob[-1]+1,:,:] = results[i][1]

rayEndTime = time()
print("Transition matrix calculation time:", rayEndTime - startTime)

#save the transition and value function mask matrix to disk
np.save('P.npy', P)
np.save('V_mask.npy', V_mask)

#save the control and state space so no need for future recomputation
np.savetxt('controlSpace.txt', discreteControlSpace)
np.savetxt('stateSpace.txt', discreteStateSpace)
