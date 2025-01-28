import ctypes
import os.path

cwd = os.path.dirname(__file__)


def kernel_function(func, blocks, threads):
	def wrapper(*args):
		expected_arguments = len(func.argtypes) # -2
		given_arguments = len(args)
		if given_arguments == expected_arguments:
			try:
				return_values = func(*args) # , blocks, threads
			except Exception as e:
				print(f"Error calling CUDA function: {e}")
			
			pythonic_returns = [return_values[i] for i in range(len(args[0]))]

			# Prevent memory leak :)
			func.freeMemory.argtypes = [ctypes.POINTER(ctypes.c_int)]
			func.freeMemory.restype = None
			func.freeMemory(return_values)
			
			return pythonic_returns
		else:
			raise TypeError(f'{func.__name__}() takes {expected_arguments} positional arguments but {given_arguments} were given')
	return wrapper


def load_kernel(path, kernel_name, blocks=1, threads=1):
	# Load the shared dll
	shared_dll = ctypes.CDLL(path)
	kernel = shared_dll.__getattr__(kernel_name)
	kernel.freeMemory = shared_dll.freeMemory
	kernel.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
	kernel.restype = ctypes.POINTER(ctypes.c_int)
	return kernel_function(kernel, blocks, threads)