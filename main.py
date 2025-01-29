import cuda_python
import os.path

cwd = os.path.dirname(__file__)

add_dll = os.path.join(cwd, "compiled_kernel.dll")
gpu_add = cuda_python.load_kernel(add_dll, 'addArrays')

a = cuda_python.convert((1,2,3), 'cuda')
b = cuda_python.convert((2,3,4), 'cuda')
c = cuda_python.convert((3,4,5), 'cuda')

d = gpu_add(a, b)

print(cuda_python.convert(d, 'python'))
# prints [6 8 10]

e = gpu_add(d, c)

print(cuda_python.convert(e, 'python'))
# prints [9 12 15]