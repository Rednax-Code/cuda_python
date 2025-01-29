"""
The base handler. Mainly for loading in kernels and decorating their behavior when called.
"""

from collections.abc import Callable
from os import path, PathLike
from ctypes import c_char_p, CDLL, POINTER
from .class_types import cIntArray
from .type_converter import convert_inout_data_types
from .signature_parser import parse

cwd = path.dirname(__file__)


def kernel_function(func):
	def wrapper(*args):
		expected_arguments = len(func.argtypes)
		given_arguments = len(args)
		if given_arguments == expected_arguments:
			try:
				return_values = func(*args)
			except Exception as e:
				print(f"Error calling CUDA function: {e}")
			
			result = return_values.contents

			# Prevent memory leak :)
			func.freeArray(return_values)
			
			return result
		else:
			raise TypeError(f'{func.__name__}() takes {expected_arguments} positional arguments but {given_arguments} were given')
	return wrapper


def load_kernel(path: str | PathLike, kernel_name: str) -> Callable:
	# Load the shared dll
	shared_dll = CDLL(path)
	kernel = shared_dll.__getattr__(kernel_name)
	
	signature_function = shared_dll.__getattr__(f'{kernel_name}Signature')
	signature_function.restype = c_char_p
	signature = signature_function().decode("utf-8")
	input_types, output_types = parse(signature)

	converted_input_types = convert_inout_data_types(input_types, 'in')
	converted_output_types = convert_inout_data_types(output_types, 'out')

	kernel.argtypes = converted_input_types
	kernel.restype = converted_output_types

	shared_dll.freeArray.argtypes = [POINTER(cIntArray)]
	shared_dll.freeArray.restype = None
	kernel.freeArray = shared_dll.freeArray	

	return kernel_function(kernel)


# Cleaning up namespace
del Callable
del path, PathLike