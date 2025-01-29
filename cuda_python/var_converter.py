"""
This converts ctypes variables to python variables and back.
"""

import numpy as np

from ctypes import c_int, c_float, c_double, POINTER, _Pointer
from typing import Literal
from .class_types import cIntArray, cFloatArray
from .error_messages import not_supported


# The stuff below is also a representation of all supported types
cuda_variable = c_int | c_double | _Pointer
python_variable = int | float | list[int] | list[float] | np.ndarray


def convert_to_cuda(variable: python_variable) -> cuda_variable:
	
	# Convert lists and arrays
	if isinstance(variable, (list, tuple, np.ndarray)):
		if isinstance(variable, (list, tuple)):
			variable = np.array(variable)
		if np.issubdtype(variable.dtype, np.integer):
			return cIntArray(variable)
		elif np.issubdtype(variable.dtype, np.floating):
			return cFloatArray(variable)
	
	# Convert numbers
	elif isinstance(variable, int):
		return c_int(variable)
	elif isinstance(variable, float):
		return c_double(variable)

	else:
		raise TypeError(not_supported.format(type(variable)))


def convert_from_cuda(variable: cuda_variable) ->  python_variable:

	# Convert lists and arrays
	if isinstance(variable, (cIntArray, cFloatArray)):
		return variable.array.copy()

	# Convert numbers
	elif isinstance(variable, c_int):
		return int(variable.value)
	elif isinstance(variable, (c_float, c_double)):
		return float(variable.value)
	
	else:
		raise TypeError(not_supported.format(type(variable)))


def convert(variable: object, target: Literal['python', 'cuda']) -> (int | float | np.ndarray | c_int | c_double | _Pointer):
	"""This function converts python variable types to/from types interpretable by cuda.

	`Cool`

	Parameters
	----------
	variable
		The variable you wish to convert.
	target
		The target, so either 'python' or 'cuda'.

	Returns
	-------
	(int | float | np.ndarray | ctypes.c_int | ctypes.c_double | ctypes._Pointer)
		The converted version of the variable
	
	Raises
	------
	ValueError
		When `target` is not equal to either 'python' or 'cuda'.
	TypeError
		When `variable` is not an instance of any supported type.
	"""
	if target == 'python':
		converted_variable = convert_from_cuda(variable)
	elif target == 'cuda':
		converted_variable = convert_to_cuda(variable)
	else:
		raise ValueError('Expected "target" to be either \'python\' or \'cuda\'')

	return converted_variable