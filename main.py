import cuda_python
import os.path

cwd = os.path.dirname(__file__)

add_dll = os.path.join(cwd, "compiled_kernel.dll")
add_inputs = [list[int], list[int]]
add_outputs = [list[int]]
gpu_add = cuda_python.load_kernel(add_dll, 'addArrays', add_inputs, add_outputs)

a = cuda_python.convert((1,0,0), 'cuda')
b = cuda_python.convert((0,0,0), 'cuda')
c = cuda_python.convert((0,0,0), 'cuda')

d = gpu_add(a, b)

print(cuda_python.convert(d, 'python'))
# prints [6 8 10]

e = gpu_add(d, c)

print(cuda_python.convert(e, 'python'))
# prints [9 12 15]