import numpy as np
from casadi import *

def getError(currentState, referenceState):
    return currentState - referenceState

def getPosition(state):
    return state[0:2]

def getOrientation(state):
    return state[2]

def errorMotionModelNoNoise(delta_t, p_err, u_t, currRefState, nextRefState):
    #u_t_2d = np.atleast_2d(u_t).T
    u_t_2d = np.array([[u_t[0]],[u_t[1]]])
    #p_err_2d = np.atleast_2d(p_err).T
    p_err_2d = np.array([[p_err[0]],[p_err[1]],[p_err[2]]])
    G = np.array([[delta_t * np.cos(p_err[2] + currRefState[2]), 0], [delta_t * np.sin(p_err[2] + currRefState[2]), 0], [0, delta_t]])
    refPosDiff = np.atleast_2d(np.array(currRefState[0:2]) - np.array(nextRefState[0:2])).T
    refOriDiff = np.atleast_2d(np.array(currRefState[2] - nextRefState[2]))
    refDiff = np.vstack((refPosDiff,refOriDiff))
    p_err_next = (p_err_2d + G @ u_t_2d + refDiff).flatten()
    print(p_err_next)
    return vertcat(p_err_next[0], p_err_next[1], p_err_next[2])
    #print(np.array([p_err_next[0,0], p_err_next[1,0], p_err_next[2,0]])) 
    #return np.array([p_err_next[0,0], p_err_next[1,0], p_err_next[2,0]])
