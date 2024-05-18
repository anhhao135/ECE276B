import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Planner
from main import *
from lib import *


boundary, blocks = load_map('./maps/my_cube.txt')
point1 = np.array([2,-3,2])
point2 = np.array([2.1,4,2])
path = np.vstack((point1, point2))
fig, ax, hb, hs, hg = draw_map(boundary, blocks, point1, point2)
ax.plot(path[:,0],path[:,1],path[:,2],'r-')
print(checkCollision(point1, point2, blocks[0]))
plt.show(block=True)

