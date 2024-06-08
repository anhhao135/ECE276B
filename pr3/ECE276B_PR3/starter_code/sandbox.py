from time import time
import numpy as np
import utils
import matplotlib.pyplot as plt


#theta error range pi to -pi

sparseThetaErrorBounds = [-np.pi, np.pi]
sparsePositionErrorBounds = [-3, 3]

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

plt.show()


