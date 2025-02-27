import numpy as np
from numpy import sin, cos, pi
from casadi import *
from utils import *
from scipy.stats import multivariate_normal
import scipy.sparse
#common functions

def getError(currentState, referenceState):
    #return error between current robot state and reference trajectory
    error = currentState - referenceState
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    return error

def errorMotionModelNoNoise(delta_t, p_err, theta_err, u_t, currRefState, nextRefState, angleWrap = False):
    #error motion model defined in project handout
    #coded to work with Casadi solver variables
    u_t_2d = np.array([[u_t[0]],[u_t[1]]])
    p_err_2d = np.array([[p_err[0]],[p_err[1]],[theta_err]])
    G = np.array([[delta_t * np.cos(theta_err + currRefState[2]), 0], [delta_t * np.sin(theta_err + currRefState[2]), 0], [0, delta_t]])
    refPosDiff = np.atleast_2d(np.array(currRefState[0:2]) - np.array(nextRefState[0:2])).T
    refOriDiff = np.atleast_2d(np.array(currRefState[2] - nextRefState[2]))
    refDiff = np.vstack((refPosDiff,refOriDiff))
    p_err_next = (p_err_2d + G @ u_t_2d + refDiff).flatten()
    if angleWrap:
        p_err_next[2] = np.arctan2(np.sin(p_err_next[2]), np.cos(p_err_next[2]))
    return vertcat(p_err_next[0], p_err_next[1], p_err_next[2])



#CEC NLP solver functions

def NLP_controller(delta_t, horizon, traj, currentIter, currentState, freeSpaceBounds, obstacle1, obstacle2, Q_in, R_in, q_in, obstaclePadding, gamma_in):
        
    #set up obstacle description variables
    obstacleCenter = np.array([[obstacle1[0], obstacle1[1]]]).T
    obstacleRadius = obstacle1[2]

    obstacleCenter2 = np.array([[obstacle2[0], obstacle2[1]]]).T
    obstacleRadius2 = obstacle2[2]

    #pre-generate reference trajectory ahead by time horizon
    referenceStatesAhead = []
    for i in range(currentIter, currentIter + horizon + 1):
        referenceStatesAhead.append(traj(i))
    referenceStatesAhead = np.array(referenceStatesAhead)

    #the below logic is to solve the issue of angle wrap around that can cause the NLP solver to produce extreme control inputs
    #consider a transition from pi to -pi caused by a slight counter clockwise rotation
    #this would shift the orientation error state suddenly by almost 2pi, causing the solver to produce extreme angular velocities to turn clockwise for the correction when in reality, we want a slight counter clockwise rotation
    #we modify the reference state or the current state orientation component such that they are part of a continuous 1D space
    referenceStatesAhead = np.array(referenceStatesAhead)
    referenceStatesAheadThetas = referenceStatesAhead[:,2]
    referenceStatesAheadThetas = np.concatenate((referenceStatesAheadThetas, np.atleast_1d(currentState[2]))) #add the current orientation state to the collection of orientations
    referenceStatesAheadThetas = np.unwrap(referenceStatesAheadThetas) #unwrap ensures all the orientations follow a continuous space
    referenceStatesAhead[:,2] = referenceStatesAheadThetas[:-1] #place continuous reference orientations back to reference poses ahead
    currentState[2] = referenceStatesAheadThetas[-1] #place continuous current orientation back to current state

    #set up NLP solver variables
    U = MX.sym('U', 2 * horizon)
    P = MX.sym('E', 2 * horizon + 2)
    Theta = MX.sym('theta', horizon + 1)
    E_given = getError(currentState, referenceStatesAhead[0]) #get the current error state

    #define the control space boundaries
    lowerBoundv = 0
    upperBoundv = 1
    lowerBoundw = -1
    upperBoundw = 1

    #define the position error boundaries i.e. we are allowing the maximum position error to be 1 in magnitude
    lowerBoundPositionError = -1
    upperBoundPositionError = 1

    #define the orientation error boundaries i.e. we are allowing the maximum orientation error to be 0.5 in magnitude
    lowerBoundOrientationError = -0.5
    upperBoundOrientationError = 0.5


    #define the constraint vectors to be fed into the Casadi NLP solver
    controlInputSolverLowerConstraint = list(np.tile(np.array([lowerBoundv, lowerBoundw]), horizon))
    controlInputSolverUpperConstraint = list(np.tile(np.array([upperBoundv, upperBoundw]), horizon))

    positionErrorSolverLowerConstraint = list(lowerBoundPositionError * np.ones(2 * horizon + 2))
    positionErrorSolverUpperConstraint = list(upperBoundPositionError * np.ones(2 * horizon + 2))

    orientationErrorSolverLowerConstraint = list(lowerBoundOrientationError * np.ones(horizon + 1))
    orientationErrorSolverUpperConstraint = list(upperBoundOrientationError * np.ones(horizon + 1))

    initialConditionSolverConstraint = list(np.zeros(5 * horizon + 3)) #initialize all variables to 0
    motionModelSolverConstraint = list(np.zeros((horizon+1) * 3))

    upperBoundObstacleAvoider = np.inf #there is no upper boundary for how far away the robot can be from the obstacle

    obstacleSolverLowerConstraint = list(np.zeros(2 * horizon))
    obstacleSolverUpperConstraint = list(upperBoundObstacleAvoider * np.ones(2 * horizon))

    #constrain the position error to the map boundaries
    mapBoundSolverLowerConstraint = list(np.tile(np.array([freeSpaceBounds[0], freeSpaceBounds[1]]), horizon))
    mapBoundSolverUpperConstraint = list(np.tile(np.array([freeSpaceBounds[2], freeSpaceBounds[3]]), horizon))


    variables = vertcat(U, P, Theta)



    #define the stage cost penalty variables in matrix form
    Q = Q_in * np.eye(2)
    QBatch = np.kron(np.eye(horizon,dtype=int),Q)

    R = R_in * np.eye(2)
    RBatch = np.kron(np.eye(horizon,dtype=int),R)

    q = q_in

    #create gammas vector and matrix
    gammaValue = gamma_in
    gammas = np.zeros(horizon)
    for i in range(horizon):
        gammas[i] = gammaValue**i

    gammas2D = np.eye(2 * horizon)
    for i in range(horizon):
        gammas2D[i*2:i*2+2,i*2:i*2+2] = gammas[i] * np.eye(2)


    #define the objective function in matrix form as the sum of the stage costs
    #the terminal cost is simply the norm squared of the last error state i.e. we want to terminate ideally with no error
    costFunction = (P[2*(horizon-1)]**2 + P[2*(horizon-1) + 1]**2 + Theta[horizon-1]**2) + P[:2*horizon].T @ gammas2D @ QBatch @ P[:2*horizon] + U.T @ gammas2D @ RBatch @ U + q * np.atleast_2d(gammas) @ (1 - cos(Theta[:horizon]))**2 

    motionModelConstraint0 = vertcat(P[0:2], Theta[0]) - E_given #ensure the first state error is equal to the current error state
    g = vertcat(motionModelConstraint0)


    for i in range(horizon):
        #define the motion model constraints
        motionModelConstraint = vertcat(P[(i+1)*2:(i+1)*2+2], Theta[i+1]) - errorMotionModelNoNoise(delta_t, P[i*2:i*2+2], Theta[i], U[i*2:i*2+2], referenceStatesAhead[i], referenceStatesAhead[i+1])
        g = vertcat(g, motionModelConstraint)

    for i in range(horizon):
        #define constraint to avoid obstacle 1
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2] - obstacleCenter
        g = vertcat(g, d.T @ d - (obstacleRadius+obstaclePadding)**2) 
    
    for i in range(horizon):
        #define constraint to avoid obstacle 2
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2] - obstacleCenter2
        g = vertcat(g, d.T @ d - (obstacleRadius2+obstaclePadding)**2) 

    for i in range(horizon):
        #define map boundary constraint
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2]
        g = vertcat(g, d)

    solver_params = {
    "ubg": motionModelSolverConstraint + obstacleSolverUpperConstraint + mapBoundSolverUpperConstraint,
    "lbg" :motionModelSolverConstraint + obstacleSolverLowerConstraint + mapBoundSolverLowerConstraint,
    "lbx": controlInputSolverLowerConstraint + positionErrorSolverLowerConstraint + orientationErrorSolverLowerConstraint,
    "ubx": controlInputSolverUpperConstraint + positionErrorSolverUpperConstraint + orientationErrorSolverUpperConstraint,
    "x0": initialConditionSolverConstraint
    }

    #run Casadi NLP solver with constraints
    opts = {'ipopt.print_level':0, 'print_time':0}
    solver = nlpsol("solver", "ipopt", {'x':variables, 'f':costFunction, 'g':g}, opts)
    sol = solver(**solver_params)

    #return the first control input of the solution sequence
    return [float(sol["x"][0]), float(sol["x"][1])]




#GPI controller functions

def GPI_controller(curTime, currentState, stateSpace, controlSpace, traj, policy):

    #this is the GPI controller function which converts the current robot state into an error state, uses that to query the policy to get the next best control input

    #the below logic is to solve the issue of angle wrap around that can cause the GPI controller to produce extreme control inputs
    #consider a transition from pi to -pi caused by a slight counter clockwise rotation
    #this would shift the orientation error state suddenly by almost 2pi, causing the solver to produce extreme angular velocities to turn clockwise for the correction when in reality, we want a slight counter clockwise rotation
    #we modify the reference state or the current state orientation component such that they are part of a continuous 1D space

    referenceStatesAhead = []
    for i in range(curTime, curTime + 5):
        referenceStatesAhead.append(traj(i))
    referenceStatesAhead = np.array(referenceStatesAhead)
    referenceStatesAheadThetas = referenceStatesAhead[:,2]
    referenceStatesAheadThetas = np.concatenate((referenceStatesAheadThetas, np.atleast_1d(currentState[2])))
    referenceStatesAheadThetas = np.unwrap(referenceStatesAheadThetas)
    referenceStatesAhead[:,2] = referenceStatesAheadThetas[:-1]
    currentState[2] = referenceStatesAheadThetas[-1]
    curTime = curTime % T
    errorState = getError(currentState, referenceStatesAhead[0])
    errorState = np.append(errorState, curTime) #the error state also has a time component

    discreteState = continuousToDiscreteState(errorState,stateSpace) #convert the error state to discrete

    return controlSpace[int(policy[discreteState])] #query the policy for the optimal control input



def constructDiscreteStateSpace(timeStepsCount = 100, positionErrorBoundMagnitude = 3, thetaErrorBoundMagnitude = np.pi, sparseDiscretizationCount = 9, densePositionErrorShrinkFactor = 0.3, denseThetaErrorShrinkFactor = 0.3):
    #construction of discrete state space
    #uses adaptive grid so space closer to origin has finer grid spacing

    sparseThetaErrorBounds = [-thetaErrorBoundMagnitude, thetaErrorBoundMagnitude] #define angular error bounds
    sparsePositionErrorBounds = [-positionErrorBoundMagnitude, positionErrorBoundMagnitude] #define position error bounds

    #create a sparse mesh grid from the bounds
    sparseXError = np.linspace(sparsePositionErrorBounds[0], sparsePositionErrorBounds[1], sparseDiscretizationCount)
    sparseYError = np.linspace(sparsePositionErrorBounds[0], sparsePositionErrorBounds[1], sparseDiscretizationCount)
    sparseThetaError = np.linspace(sparseThetaErrorBounds[0], sparseThetaErrorBounds[1], sparseDiscretizationCount)
    sparseXErrorGrid, sparseYErrorGrid, sparseZErrorGrid = np.meshgrid(sparseXError, sparseYError, sparseThetaError, indexing='ij')



    #create a dense mesh grid from the shrunken bounds
    denseThetaErrorBounds = [denseThetaErrorShrinkFactor * sparseThetaErrorBounds[0], denseThetaErrorShrinkFactor * sparseThetaErrorBounds[1]]
    densePositionErrorBounds = [densePositionErrorShrinkFactor * sparsePositionErrorBounds[0], densePositionErrorShrinkFactor * sparsePositionErrorBounds[1]]

    denseDiscretizationCount = sparseDiscretizationCount

    denseXError = np.linspace(densePositionErrorBounds[0], densePositionErrorBounds[1], denseDiscretizationCount)
    denseYError = np.linspace(densePositionErrorBounds[0], densePositionErrorBounds[1], denseDiscretizationCount)
    denseThetaError = np.linspace(denseThetaErrorBounds[0], denseThetaErrorBounds[1], denseDiscretizationCount)
    denseXErrorGrid, denseYErrorGrid, denseZErrorGrid = np.meshgrid(denseXError, denseYError, denseThetaError, indexing='ij')


    #now we check for any overlap in the two meshgrids when merging them
    #effectively performing a union
    discreteStateSpace = []

    for t in range(timeStepsCount):
        for i in range(denseDiscretizationCount):
            for j in range(denseDiscretizationCount):
                for k in range(denseDiscretizationCount):
                    x = denseXErrorGrid[i,j,k]
                    y = denseYErrorGrid[i,j,k]
                    z = denseZErrorGrid[i,j,k]
                    discreteStateSpace.append([x,y,z,t])
        for i in range(sparseDiscretizationCount):
            for j in range(sparseDiscretizationCount):
                for k in range(sparseDiscretizationCount):
                    x = sparseXErrorGrid[i,j,k]
                    y = sparseYErrorGrid[i,j,k]
                    z = sparseZErrorGrid[i,j,k]
                    if not (x > densePositionErrorBounds[0] and x < densePositionErrorBounds[1] and y > densePositionErrorBounds[0] and y < densePositionErrorBounds[1] and z > denseThetaErrorBounds[0] and z < denseThetaErrorBounds[1]):
                        discreteStateSpace.append([x,y,z,t])

    return discreteStateSpace

def constructDiscreteControlSpace(controlVUpperBound = 1, controlVLowerBound = 0, controlWBoundMagnitude = 1, sparseControlDiscretizationCount = 5, densesControlVShrinkFactor = 0.3, denseControlWShrinkFactor = 0.3):

    #construction of discrete control space
    #uses adaptive grid so space closer to origin has finer grid spacing
    #see function for state space discretization for more comments

    sparseControlVBounds = [controlVLowerBound, controlVUpperBound]
    sparseControlWBounds = [-controlWBoundMagnitude, controlWBoundMagnitude]

    sparseControlV = np.linspace(sparseControlVBounds[0], sparseControlVBounds[1], sparseControlDiscretizationCount)
    sparseControlW = np.linspace(sparseControlWBounds[0], sparseControlWBounds[1], sparseControlDiscretizationCount)
    sparseControlVGrid, sparseControlWGrid = np.meshgrid(sparseControlV, sparseControlW, indexing='ij')

    denseControlVBounds = [densesControlVShrinkFactor * sparseControlVBounds[0], densesControlVShrinkFactor * sparseControlVBounds[1]]
    denseControlWBounds = [denseControlWShrinkFactor * sparseControlWBounds[0], denseControlWShrinkFactor * sparseControlWBounds[1]]

    denseControlDiscretizationCount = sparseControlDiscretizationCount

    denseControlV = np.linspace(denseControlVBounds[0], denseControlVBounds[1], denseControlDiscretizationCount)
    denseControlW = np.linspace(denseControlWBounds[0], denseControlWBounds[1], denseControlDiscretizationCount)
    denseControlVGrid, denseControlWGrid = np.meshgrid(denseControlV, denseControlW, indexing='ij')


    discreteControlSpace = []

    for i in range(denseControlDiscretizationCount):
        for j in range(denseControlDiscretizationCount):
            x = denseControlVGrid[i,j]
            y = denseControlWGrid[i,j]
            discreteControlSpace.append([x,y])

    for i in range(sparseControlDiscretizationCount):
        for j in range(sparseControlDiscretizationCount):
            x = sparseControlVGrid[i,j]
            y = sparseControlWGrid[i,j]
            if not (x > denseControlVBounds[0] and x < denseControlVBounds[1] and y > denseControlWBounds[0] and y < denseControlWBounds[1]):
                discreteControlSpace.append([x,y])

    return discreteControlSpace


def continuousToDiscreteStateAndNeighbors(continuousState, discreteStates, neighborCount):
    #convert a continuous state to a discrete state index, along with the indexes of its neighbors
    distances = continuousState[0:3] - discreteStates[:, 0:3]
    distances = np.linalg.norm(distances, axis=1)
    smallestDistances = distances.argsort()[:neighborCount+1]
    return smallestDistances

def continuousToDiscreteState(continuousState, discreteStates):
    #convert a continuous state to a discrete state index, along with the indexes of its neighbors
    distances = continuousState - discreteStates
    distances = np.linalg.norm(distances, axis=1)
    return np.argmin(distances)


def findNeighborsOfState(discreteStates, perTimeStateSize, numberOfNeighbors):
    #input, discrete state space
    #output, a matrix to lookup the stochastic vector of neighbors of the state and their likelihoods, given that input state is the mean
    states = discreteStates[0:perTimeStateSize, 0:3] #neighbors are not time variant so we do not need the time component
    perTimeStateNeighbors = np.zeros((perTimeStateSize, perTimeStateSize))
    for i in range(perTimeStateSize):
        neighborStatesIndexes = continuousToDiscreteStateAndNeighbors(states[i], states, numberOfNeighbors)
        neighborStates = states[neighborStatesIndexes]
        likelihoods = multivariate_normal.pdf(neighborStates, states[i], sigma) #get the likelihood from the gaussian pdf
        likelihoodsNormalized = likelihoods / np.sum(likelihoods, axis=0) #normalize the likelihoods so they sum to 1
        likelihoodsVector = np.zeros(perTimeStateSize)
        likelihoodsVector[neighborStatesIndexes.astype(int)] = likelihoodsNormalized
        perTimeStateNeighbors[i,:] = likelihoodsVector
    return perTimeStateNeighbors




def GPI_CalculatePolicy(Q_in, R_in, q_in, gamma_in, traj, obstaclePenalty, obstacle1, obstacle2):
    #obstacle1, obstacle2,
    #script to calculate the optimal policy using a precomputed transition matrix for GPI
    #specify tuning parameters here


    #load state and control space
    discreteStateSpace = np.loadtxt('stateSpace.txt')
    discreteControlSpace = np.loadtxt('controlSpace.txt')
    stateSpaceSize = discreteStateSpace.shape[0]
    perTimeStateSpaceSize = int(stateSpaceSize / T)
    controlSpaceSize = discreteControlSpace.shape[0]

    #tune GPI parameters here

    Q = Q_in * scipy.sparse.eye(2)
    R = R_in * scipy.sparse.eye(2)
    q = q_in
    gamma = gamma_in


    #Q = 8 * scipy.sparse.eye(2)
    #R = 1 * scipy.sparse.eye(2)
    #q = 10
    #gamma = 0.95


    #specify the amount of GPI iterations
    iterations = 200

    #specify convergence threshold for change in value function:
    terminationThresh = 0.00001



    #construction of the stage cost in matrix form

    L = np.zeros((stateSpaceSize,controlSpaceSize))

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


    refPositions = np.zeros((stateSpaceSize, 2))
    for i in range(0, 100):
        refPositions[i * perTimeStateSpaceSize : i * perTimeStateSpaceSize + perTimeStateSpaceSize] = traj(i)[0:2]
    statePositions = discreteStateSpace[:,0:2] + refPositions

    obstacleCenter = np.array([[obstacle1[0], obstacle1[1]]]).T
    obstacleRadius = obstacle1[2]
    obstacleCenter2 = np.array([[obstacle2[0], obstacle2[1]]]).T
    obstacleRadius2 = obstacle2[2]

    obstacle1Distances = np.linalg.norm(statePositions - obstacleCenter.T, axis=1)
    obstacle1TooClose = obstaclePenalty * (obstacle1Distances < obstacleRadius).astype(int)
    obstacle2Distances = np.linalg.norm(statePositions - obstacleCenter2.T, axis=1)
    obstacle2TooClose = obstaclePenalty * (obstacle2Distances < obstacleRadius2).astype(int)



    L = np.tile(L_P_err, controlSpaceSize) + np.tile(L_Theta_err, controlSpaceSize) + np.tile(L_U, stateSpaceSize).T + np.tile(np.atleast_2d(obstacle1TooClose).T, controlSpaceSize) + np.tile(np.atleast_2d(obstacle2TooClose).T, controlSpaceSize)

    #load the precomputed transition and value mask matrices
    V_mask = np.load('V_mask.npy')
    P = np.load('P.npy')

    #initialize zero value function and policy matrices
    V = np.zeros((iterations+1, stateSpaceSize))
    pi = np.zeros((iterations+1,stateSpaceSize),dtype=np.uint16)

    #core of GPI algorithm
    for k in range(iterations):
        Q = np.zeros((stateSpaceSize,controlSpaceSize))
        V_k = V[k,:][V_mask]
        Q = L + gamma * np.sum(np.multiply(P,V_k), axis=2) #calculate Q-value matrix
        pi[k+1,:] = np.argmin(Q, axis=1) #policy improvement
        V[k+1,:] = np.min(Q, axis=1) #policy evaluation; but in this case we only do it once; as long as GPI runs forever, the value function should converge
        maxValueDifference = np.abs(V[k+1] - V[k]).max() #check for the max change in the value function; bigger changes means the policy is still evolving and has not converged yet

        if maxValueDifference < terminationThresh:
            break #terminate once the policy converges sufficiently

        print("Max value function difference", maxValueDifference)


    #save policy to disk for querying
    np.savetxt('policy.txt', pi[-1])
    return pi[-1]
