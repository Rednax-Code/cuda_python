"""
The base handler. Mainly for loading in kernels and decorating their behavior when called.
"""

from collections.abc import Callable
from os import path, PathLike
from time import perf_counter
from ctypes import c_char_p, CDLL, POINTER
from .class_types import cIntArray
from .type_converter import convert_inout_data_types
from .signature_parser import parse
from .error_messages import wrong_arguments, arrays_not_same_size


cwd = path.dirname(__file__)


def c_function(func: Callable):
	def wrapper(*args, profile: bool=False):
		expected_arguments = len(func.argtypes)
		given_arguments = len(args)
		if given_arguments == expected_arguments:
			try:
				if profile:
					start_time = perf_counter()
					return_values = func(*args)
					end_time = perf_counter()
					return end_time - start_time
				else:
					return_values = func(*args)
			except Exception as e:
				print(f"Error calling CUDA function: {e}")

			# Check for nullptr
			if not return_values:
				print(return_values)
				raise ValueError(arrays_not_same_size)
			
			contents = return_values.contents
			result = contents.deep_copy()

			# Prevent memory leak :)
			func.freeArray(return_values)
			
			return result
		else:
			raise TypeError(wrong_arguments.format(func.__name__, expected_arguments, given_arguments))
	return wrapper


def load_function(path: str | PathLike, function_name: str) -> Callable:
	# Load the shared dll
	shared_dll = CDLL(path)
	function = shared_dll.__getattr__(function_name)
	
	signature_function = shared_dll.__getattr__(f'{function_name}Signature')
	signature_function.restype = c_char_p
	signature = signature_function().decode("utf-8")
	input_types, output_types = parse(signature)

	converted_input_types = convert_inout_data_types(input_types, 'in')
	converted_output_types = convert_inout_data_types(output_types, 'out')

	function.argtypes = converted_input_types
	function.restype = converted_output_types

	shared_dll.freeArray.argtypes = [POINTER(cIntArray)]
	shared_dll.freeArray.restype = None
	function.freeArray = shared_dll.freeArray

	return c_function(function)


# Cleaning up namespace
del Callable
del path, PathLike