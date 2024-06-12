# Python program to 
# demonstrate speed comparison
# between cupy and numpy
 
# Importing modules
import cupy as cp
import numpy as np
import time

print(cp.cuda.runtime.getDeviceCount())
 
# CuPy and GPU Runtime
s = time.time()
x_gpu = cp.ones((1000, 1000, 1000))
for i in range(10):
    x_gpu = cp.ones((1000, 1000, 1000))
    print(i)
e = time.time()
print("\nTime consumed by cupy: ", e - s)