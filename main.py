import cuda_python
import os.path
import numpy as np
import ctypes

cwd = os.path.dirname(__file__)

test_kernel = os.path.join(cwd, "compiled_kernel.dll")
gpu_add = cuda_python.load_kernel(test_kernel, 'addArrays')

a = (ctypes.c_int * 3)(3,2,1)
b = (ctypes.c_int * 3)(1,2,3)

c = gpu_add(a, b, len(a))

print(c)