import cuda_python
import os.path
import numpy as np
from time import perf_counter

cwd = os.path.dirname(__file__)

add_dll = os.path.join(cwd, "gpu_tester.dll")
gpu_add = cuda_python.load_kernel(add_dll, 'arrayTest')

n = 10
width, height = 200, 100 # 480, 360
size = width*height

cpu_a = np.random.randint(-10, 11, size)
cpu_b = np.random.randint(-10, 11, size)

gpu_a = cuda_python.convert(np.random.randint(-10, 11, size*n), 'cuda')
gpu_b = cuda_python.convert(np.random.randint(-10, 11, size*n), 'cuda')

gpu_add(gpu_a, gpu_b) # Running once so that the kernel gets optimized by the driver.


def cpu_add(a, b):
	for i in range(len(a)):
		s = a.sum()
	return a * b + s

start_time = perf_counter()
for i in range(n):
	start_time = perf_counter()
	return_values = cpu_add(cpu_a, cpu_b)
end_time = perf_counter()
time_cpu = end_time - start_time

time_gpu = gpu_add(gpu_a, gpu_b, profile=True)


print(f'CPU time: {time_cpu:.7f}s')
print(f'GPU time: {time_gpu:.7f}s')