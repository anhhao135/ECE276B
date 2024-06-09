import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from tqdm import tqdm
from casadi import *
from utils import *



def circle(k):
    a = 2 * np.pi / (T * time_step)
    delta = np.pi / 2
    radius = 1.5
    xref_start = 0
    yref_start = 0
    k = k % T
    xref = xref_start + radius * np.cos(a * k * time_step + delta)
    yref = yref_start + radius * np.sin(a * k * time_step + delta)
    thetaref = np.arctan2(yref, xref) + np.pi/2
    return [xref, yref, thetaref]

def getError(currentState, referenceState):
    return currentState - referenceState

def getPosition(state):
    return state[0:2]

def getOrientation(state):
    return state[2]

def errorMotionModelNoNoise(delta_t, p_err, theta_err, u_t, currRefState, nextRefState):
    u_t_2d = np.array([[u_t[0]],[u_t[1]]])
    p_err_2d = np.array([[p_err[0]],[p_err[1]],[theta_err]])
    G = np.array([[delta_t * np.cos(theta_err + currRefState[2]), 0], [delta_t * np.sin(theta_err + currRefState[2]), 0], [0, delta_t]])
    refPosDiff = np.atleast_2d(np.array(currRefState[0:2]) - np.array(nextRefState[0:2])).T
    refOriDiff = np.atleast_2d(np.array(currRefState[2] - nextRefState[2]))
    refDiff = np.vstack((refPosDiff,refOriDiff))
    p_err_next = (p_err_2d + G @ u_t_2d + refDiff).flatten()
    return vertcat(p_err_next[0], p_err_next[1], p_err_next[2])



def NLP_controller(delta_t, horizon, traj, currentIter, currentState, freeSpaceBounds, obstacle1, obstacle2, obstaclePadding):

    obstacleCenter = np.array([[obstacle1[0], obstacle1[1]]]).T
    obstacleRadius = obstacle1[2]

    obstacleCenter2 = np.array([[obstacle2[0], obstacle2[1]]]).T
    obstacleRadius2 = obstacle2[2]

    referenceStatesAhead = []

    for i in range(currentIter, currentIter + horizon + 1):
        referenceStatesAhead.append(traj(i))


    U = MX.sym('U', 2 * horizon)
    P = MX.sym('E', 2 * horizon + 2)
    Theta = MX.sym('theta', horizon + 1)
    E_given = getError(currentState, referenceStatesAhead[0])

    lowerBoundv = 0
    upperBoundv = 10
    lowerBoundw = -10
    upperBoundw = 10

    lowerBoundPositionError = -5
    upperBoundPositionError = 5

    lowerBoundOrientationError = -0.3
    upperBoundOrientationError = 0.3

    controlInputSolverLowerConstraint = list(np.tile(np.array([lowerBoundv, lowerBoundw]), horizon))
    controlInputSolverUpperConstraint = list(np.tile(np.array([upperBoundv, upperBoundw]), horizon))

    positionErrorSolverLowerConstraint = list(lowerBoundPositionError * np.ones(2 * horizon + 2))
    positionErrorSolverUpperConstraint = list(upperBoundPositionError * np.ones(2 * horizon + 2))

    orientationErrorSolverLowerConstraint = list(lowerBoundOrientationError * np.ones(horizon + 1))
    orientationErrorSolverUpperConstraint = list(upperBoundOrientationError * np.ones(horizon + 1))

    initialConditionSolverConstraint = list(np.zeros(5 * horizon + 3))
    motionModelSolverConstraint = list(np.zeros((horizon+1) * 3))


    upperBoundObstacleAvoider = 100

    obstacleSolverLowerConstraint = list(np.zeros(2 * horizon))
    obstacleSolverUpperConstraint = list(upperBoundObstacleAvoider * np.ones(2 * horizon))


    mapBoundSolverLowerConstraint = list(np.tile(np.array([freeSpaceBounds[0], freeSpaceBounds[1]]), horizon))
    mapBoundSolverUpperConstraint = list(np.tile(np.array([freeSpaceBounds[2], freeSpaceBounds[3]]), horizon))


    variables = vertcat(U, P, Theta)

    Q = 4 * np.eye(2)
    QBatch = np.kron(np.eye(horizon,dtype=int),Q)

    R = 2 * np.eye(2)
    RBatch = np.kron(np.eye(horizon,dtype=int),R)

    q = 1

    gammaValue = 0.95
    gammas = np.zeros(horizon)
    for i in range(horizon):
        gammas[i] = gammaValue**i

    gammas2D = np.eye(2 * horizon)
    for i in range(horizon):
        gammas2D[i*2:i*2+2,i*2:i*2+2] = gammas[i] * np.eye(2)


    costFunction = (P[2*(horizon-1)]**2 + P[2*(horizon-1) + 1]**2 + Theta[horizon-1]**2) + P[:2*horizon].T @ gammas2D @ QBatch @ P[:2*horizon] + U.T @ gammas2D @ RBatch @ U + q * np.atleast_2d(gammas) @ (1 - cos(Theta[:horizon]))**2

    motionModelConstraint0 = vertcat(P[0:2], Theta[0]) - E_given

    g = vertcat(motionModelConstraint0)

    for i in range(horizon):
        motionModelConstraint = vertcat(P[(i+1)*2:(i+1)*2+2], Theta[i+1]) - errorMotionModelNoNoise(delta_t, P[i*2:i*2+2], Theta[i], U[i*2:i*2+2], referenceStatesAhead[i], referenceStatesAhead[i+1])
        g = vertcat(g, motionModelConstraint)

    for i in range(horizon): #obstacle 1
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2] - obstacleCenter
        g = vertcat(g, d.T @ d - (obstacleRadius+obstaclePadding)**2) 
    
    for i in range(horizon): #obstacle 2
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2] - obstacleCenter2
        g = vertcat(g, d.T @ d - (obstacleRadius2+obstaclePadding)**2) 

    for i in range(horizon): #map bounds
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2]
        g = vertcat(g, d)

    solver_params = {
    "ubg": motionModelSolverConstraint + obstacleSolverUpperConstraint + mapBoundSolverUpperConstraint,
    "lbg" :motionModelSolverConstraint + obstacleSolverLowerConstraint + mapBoundSolverLowerConstraint,
    "lbx": controlInputSolverLowerConstraint + positionErrorSolverLowerConstraint + orientationErrorSolverLowerConstraint,
    "ubx": controlInputSolverUpperConstraint + positionErrorSolverUpperConstraint + orientationErrorSolverUpperConstraint,
    "x0": initialConditionSolverConstraint
    }

    opts = {'ipopt.print_level':0, 'print_time':0}
    solver = nlpsol("solver", "ipopt", {'x':variables, 'f':costFunction, 'g':g}, opts)


    sol = solver(**solver_params)
    return [float(sol["x"][0]), float(sol["x"][1])]



def constructDiscreteStateSpace(timeStepsCount = 100, positionErrorBoundMagnitude = 3, thetaErrorBoundMagnitude = np.pi, sparseDiscretizationCount = 7, densePositionErrorShrinkFactor = 0.3, denseThetaErrorShrinkFactor = 0.3):
    #construction of discrete state space

    sparseThetaErrorBounds = [-thetaErrorBoundMagnitude, thetaErrorBoundMagnitude]
    sparsePositionErrorBounds = [-positionErrorBoundMagnitude, positionErrorBoundMagnitude]

    sparseXError = np.linspace(sparsePositionErrorBounds[0], sparsePositionErrorBounds[1], sparseDiscretizationCount)
    sparseYError = np.linspace(sparsePositionErrorBounds[0], sparsePositionErrorBounds[1], sparseDiscretizationCount)
    sparseThetaError = np.linspace(sparseThetaErrorBounds[0], sparseThetaErrorBounds[1], sparseDiscretizationCount)
    sparseXErrorGrid, sparseYErrorGrid, sparseZErrorGrid = np.meshgrid(sparseXError, sparseYError, sparseThetaError, indexing='ij')


    denseThetaErrorBounds = [denseThetaErrorShrinkFactor * sparseThetaErrorBounds[0], denseThetaErrorShrinkFactor * sparseThetaErrorBounds[1]]
    densePositionErrorBounds = [densePositionErrorShrinkFactor * sparsePositionErrorBounds[0], densePositionErrorShrinkFactor * sparsePositionErrorBounds[1]]

    denseDiscretizationCount = sparseDiscretizationCount

    denseXError = np.linspace(densePositionErrorBounds[0], densePositionErrorBounds[1], denseDiscretizationCount)
    denseYError = np.linspace(densePositionErrorBounds[0], densePositionErrorBounds[1], denseDiscretizationCount)
    denseThetaError = np.linspace(denseThetaErrorBounds[0], denseThetaErrorBounds[1], denseDiscretizationCount)
    denseXErrorGrid, denseYErrorGrid, denseZErrorGrid = np.meshgrid(denseXError, denseYError, denseThetaError, indexing='ij')


    discreteStateSpace = []

    for i in range(denseDiscretizationCount):
        for j in range(denseDiscretizationCount):
            for k in range(denseDiscretizationCount):
                for t in range(timeStepsCount):
                    x = denseXErrorGrid[i,j,k]
                    y = denseYErrorGrid[i,j,k]
                    z = denseZErrorGrid[i,j,k]
                    discreteStateSpace.append([x,y,z,t])

    for i in range(sparseDiscretizationCount):
        for j in range(sparseDiscretizationCount):
            for k in range(sparseDiscretizationCount):
                for t in range(timeStepsCount):
                    x = sparseXErrorGrid[i,j,k]
                    y = sparseYErrorGrid[i,j,k]
                    z = sparseZErrorGrid[i,j,k]
                    if not (x > densePositionErrorBounds[0] and x < densePositionErrorBounds[1] and y > densePositionErrorBounds[0] and y < densePositionErrorBounds[1] and z > denseThetaErrorBounds[0] and z < denseThetaErrorBounds[1]):
                        discreteStateSpace.append([x,y,z,t])

    return discreteStateSpace

def constructDiscreteControlSpace(controlVBoundMagnitude = 10, controlWBoundMagnitude = 2, sparseControlDiscretizationCount = 7, densesControlVShrinkFactor = 0.3, denseControlWShrinkFactor = 0.3):

    #construction of discrete control space
    sparseControlVBounds = [-controlVBoundMagnitude, controlVBoundMagnitude]
    sparseControlWBounds = [-controlWBoundMagnitude, controlWBoundMagnitude]

     # number of grid points in one axis
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


def continuousToDiscreteState(continuousState, discreteStates):
    distances = continuousState - discreteStates
    distances = np.linalg.norm(distances, axis=1)
    return np.argmin(distances)

def getNeighboringStates(state, discreteStates, neighborCount):
    distances = state - discreteStates
    distances = np.linalg.norm(distances, axis=1)
    smallestDistances = distances.argsort()[:neighborCount]
    return smallestDistances

