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



#construction of discrete state space
discreteStateSpace = np.loadtxt('stateSpace.txt')
discreteControlSpace = np.loadtxt('controlSpace.txt')


stateSpaceSize = discreteStateSpace.shape[0]
perTimeStateSpaceSize = int(stateSpaceSize / utils.T)
controlSpaceSize = discreteControlSpace.shape[0]

print("state space size:", stateSpaceSize)
print("control space size:", controlSpaceSize)
print("per time state space shape:", perTimeStateSpaceSize)

L = np.zeros((stateSpaceSize,controlSpaceSize))
Q = 15 * scipy.sparse.eye(2)
R = 1 * scipy.sparse.eye(2)
q = 15


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

traj = utils.lissajous

iterations = 200

V_mask = np.load('V_mask.npy')
P = np.load('P.npy')

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
