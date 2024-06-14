import numpy as np
import utils
from my_utils import *
import scipy.sparse

#script to calculate the optimal policy using a precomputed transition matrix for GPI
#specify tuning parameters here


#load state and control space
discreteStateSpace = np.loadtxt('stateSpace.txt')
discreteControlSpace = np.loadtxt('controlSpace.txt')
stateSpaceSize = discreteStateSpace.shape[0]
perTimeStateSpaceSize = int(stateSpaceSize / utils.T)
controlSpaceSize = discreteControlSpace.shape[0]

#tune GPI parameters here
L = np.zeros((stateSpaceSize,controlSpaceSize))
Q = 8 * scipy.sparse.eye(2)
R = 1 * scipy.sparse.eye(2)
q = 10
gamma = 0.95


#specify the amount of GPI iterations
iterations = 300

#specify convergence threshold for change in value function:
terminationThresh = 0.0001



#construction of the stage cost in matrix form

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
    Q = L + gamma * np.sum(np.multiply(P,V_k), axis=2) #caluate Q-value matrix
    pi[k+1,:] = np.argmin(Q, axis=1) #policy improvement
    V[k+1,:] = np.min(Q, axis=1) #policy evaluation; but in this case we only do it once; as long as GPI runs forever, the value function should converge
    maxValueDifference = np.abs(V[k+1] - V[k]).max() #check for the max change in the value function; bigger changes means the policy is still evolving and has not converged yet
    discreteState = continuousToDiscreteState(np.array([0,0,0,0]),discreteStateSpace)

    if maxValueDifference < terminationThresh:
        break #terminate once the policy converges sufficiently

    print("Max value function difference", maxValueDifference)


#save policy to disk for querying
np.savetxt('policy.txt', pi[-1:])
