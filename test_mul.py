#!/usr/bin/env python3

import pycumath
import numpy as np
import time

size = 10000000
arr1 = np.linspace(1.0,100.0, size)
arr2 = np.linspace(1.0,100.0, size)

runs = 10
factor = 3.0

t0 = time.time()
for _ in range(runs):
    pycumath.PycuMath.mul(arr1, factor)
print("gpu time: {}".format(time.time()-t0))
t0 = time.time()
for _ in range(runs):
    arr2 = arr2 * factor
print("cpu time: {}".format(time.time()-t0))

print("results match: {}".format(np.allclose(arr1,arr2)))
