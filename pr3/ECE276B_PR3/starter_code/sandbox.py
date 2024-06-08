from time import time
import numpy as np
import utils
import matplotlib.pyplot as plt
from my_utils import *


#construction of discrete state space
discreteStateSpace = np.array(constructDiscreteStateSpace())
discreteControlSpace = constructDiscreteControlSpace()

exampleStateIndex = 100
exampleTargetState = np.array([0.7002, -0.7003, -0.47, 8])

matchedIndex = continuousToDiscreteState(exampleTargetState, discreteStateSpace)
print(matchedIndex)
print(discreteStateSpace[matchedIndex])