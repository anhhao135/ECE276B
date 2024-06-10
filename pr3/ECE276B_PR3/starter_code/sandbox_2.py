import numpy as np

# Read the array from disk
new_data = np.loadtxt('policy.txt')

# Note that this returned a 2D array!
print(new_data.shape)