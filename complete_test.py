import cuda_python
import os.path
import numpy as np

cwd = os.path.dirname(__file__)

test_kernel = os.path.join(cwd, "compiled_kernel.dll")
gpu_add = cuda_python.load_kernel(test_kernel, 'addArrays')

a = 1
b = 1.0
c = np.array([1, 2, 3])
d = (2, 3, 4)
e = [3, 4, 5]

A = cuda_python.convert(a, 'cuda')
B = cuda_python.convert(b, 'cuda')
C = cuda_python.convert(c, 'cuda')
D = cuda_python.convert(d, 'cuda')
E = cuda_python.convert(e, 'cuda')

c = gpu_add(C, D)

print(cuda_python.convert(c, 'python'))