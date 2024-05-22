import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from main import *
from lib import *
from pqdict import pqdict


fig = plt.figure()
ax = plt.axes(projection='3d')

point1 = np.array([1,1,1])
point2 = np.array([-6,-7,-8])

pointSteer = steerToNode(point1, point2, 50)

ax.scatter3D(point1[0], point1[1], point1[2], c='red')
ax.scatter3D(point2[0], point2[1], point2[2], c='green')
ax.scatter3D(pointSteer[0], pointSteer[1], pointSteer[2], c='blue')

print(np.linalg.norm(point1 - pointSteer))

plt.show(block=True)

