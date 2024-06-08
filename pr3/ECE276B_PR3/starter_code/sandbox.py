from time import time
import numpy as np
import utils
import matplotlib.pyplot as plt


#construction of discrete state space

positionErrorBoundMagnitude = 3

sparseThetaErrorBounds = [-np.pi, np.pi]
sparsePositionErrorBounds = [-positionErrorBoundMagnitude, positionErrorBoundMagnitude]

sparseDiscretizationCount = 10 # number of grid points in one axis

sparseXError = np.linspace(sparsePositionErrorBounds[0], sparsePositionErrorBounds[1], sparseDiscretizationCount)
sparseYError = np.linspace(sparsePositionErrorBounds[0], sparsePositionErrorBounds[1], sparseDiscretizationCount)
sparseThetaError = np.linspace(sparseThetaErrorBounds[0], sparseThetaErrorBounds[1], sparseDiscretizationCount)
sparseXErrorGrid, sparseYErrorGrid, sparseZErrorGrid = np.meshgrid(sparseXError, sparseYError, sparseThetaError, indexing='ij')


densePositionErrorShrinkFactor = 0.3
denseThetaErrorShrinkFactor = 0.3

denseThetaErrorBounds = [denseThetaErrorShrinkFactor * sparseThetaErrorBounds[0], denseThetaErrorShrinkFactor * sparseThetaErrorBounds[1]]
densePositionErrorBounds = [densePositionErrorShrinkFactor * sparsePositionErrorBounds[0], densePositionErrorShrinkFactor * sparsePositionErrorBounds[1]]

denseDiscretizationCount = sparseDiscretizationCount

denseXError = np.linspace(densePositionErrorBounds[0], densePositionErrorBounds[1], denseDiscretizationCount)
denseYError = np.linspace(densePositionErrorBounds[0], densePositionErrorBounds[1], denseDiscretizationCount)
denseThetaError = np.linspace(denseThetaErrorBounds[0], denseThetaErrorBounds[1], denseDiscretizationCount)
denseXErrorGrid, denseYErrorGrid, denseZErrorGrid = np.meshgrid(denseXError, denseYError, denseThetaError, indexing='ij')


discreteStateSpace = []


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')


for i in range(denseDiscretizationCount):
    for j in range(denseDiscretizationCount):
        for k in range(denseDiscretizationCount):
            x = denseXErrorGrid[i,j,k]
            y = denseYErrorGrid[i,j,k]
            z = denseZErrorGrid[i,j,k]
            discreteStateSpace.append([x,y,z])
            #ax.scatter(x, y, z, marker='o', color='green')

for i in range(sparseDiscretizationCount):
    for j in range(sparseDiscretizationCount):
        for k in range(sparseDiscretizationCount):
            x = sparseXErrorGrid[i,j,k]
            y = sparseYErrorGrid[i,j,k]
            z = sparseZErrorGrid[i,j,k]
            if not (x > densePositionErrorBounds[0] and x < densePositionErrorBounds[1] and y > densePositionErrorBounds[0] and y < densePositionErrorBounds[1] and z > denseThetaErrorBounds[0] and z < denseThetaErrorBounds[1]):
                discreteStateSpace.append([x,y,z])
                #ax.scatter(x, y, z, marker='o', color='red')


print(len(discreteStateSpace))
#plt.show()




#construction of discrete control space
controlVBoundMagnitude = 10
controlWBoundMagnitude = 2 

sparseControlVBounds = [-controlVBoundMagnitude, controlVBoundMagnitude]
sparseControlWBounds = [-controlWBoundMagnitude, controlWBoundMagnitude]

sparseControlDiscretizationCount = 7 # number of grid points in one axis

sparseControlV = np.linspace(sparseControlVBounds[0], sparseControlVBounds[1], sparseControlDiscretizationCount)
sparseControlW = np.linspace(sparseControlWBounds[0], sparseControlWBounds[1], sparseControlDiscretizationCount)
sparseControlVGrid, sparseControlWGrid = np.meshgrid(sparseControlV, sparseControlW, indexing='ij')


densesControlVShrinkFactor = 0.3
denseControlWShrinkFactor = 0.3

denseControlVBounds = [densesControlVShrinkFactor * sparseControlVBounds[0], densesControlVShrinkFactor * sparseControlVBounds[1]]
denseControlWBounds = [denseControlWShrinkFactor * sparseControlWBounds[0], denseControlWShrinkFactor * sparseControlWBounds[1]]

denseControlDiscretizationCount = sparseControlDiscretizationCount

denseControlV = np.linspace(denseControlVBounds[0], denseControlVBounds[1], denseControlDiscretizationCount)
denseControlW = np.linspace(denseControlWBounds[0], denseControlWBounds[1], denseControlDiscretizationCount)
denseControlVGrid, denseControlWGrid = np.meshgrid(denseControlV, denseControlW, indexing='ij')


discreteControlSpace = []

fig = plt.figure()
ax = fig.add_subplot()

for i in range(denseControlDiscretizationCount):
    for j in range(denseControlDiscretizationCount):
        x = denseControlVGrid[i,j]
        y = denseControlWGrid[i,j]
        discreteControlSpace.append([x,y])
        ax.scatter(x, y, marker='o', color='green')

for i in range(sparseControlDiscretizationCount):
    for j in range(sparseControlDiscretizationCount):
        x = sparseControlVGrid[i,j]
        y = sparseControlWGrid[i,j]
        if not (x > denseControlVBounds[0] and x < denseControlVBounds[1] and y > denseControlWBounds[0] and y < denseControlWBounds[1]):
            discreteControlSpace.append([x,y])
            ax.scatter(x, y, marker='o', color='red')


print(len(discreteControlSpace))
plt.show()




