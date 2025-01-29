import cuda_python
import os.path
import numpy as np

cwd = os.path.dirname(__file__)

test_kernel = os.path.join(cwd, "compiled_kernel.dll")
gpu_add = cuda_python.load_kernel(test_kernel, 'addArrays')


Array = cuda_python.convert(np.array([0, 2, 3]), 'cuda')
Tuple = cuda_python.convert((2, 3, 4), 'cuda')
List = cuda_python.convert([1, 4, 5], 'cuda')

Array_List = gpu_add(Array, List)
#G = gpu_add(F, E)

print(cuda_python.convert(Array_List, 'python'))
#print(cuda_python.convert(G, 'python'))