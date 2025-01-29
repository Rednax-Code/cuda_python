from numpy import array
from ctypes import c_int, c_double, POINTER
from typing import Literal
from .class_types import cIntArray, cFloatArray
from .error_messages import specify_type

# This is used when you load a new kernel
type_conversions = {
	int: c_int,
	float: c_double,
	list[int]: cIntArray,
	list[float]: cFloatArray
}

def convert_inout_data_types(data_types: list[object], in_or_out: Literal['in', 'out']) -> list[object]:
	# This converts data types from pythonic to ctypes.
	
	converted_types = []
	for item in data_types:
		if item == list:
			raise ValueError(specify_type)
		
		if in_or_out == 'in':
			converted_types.append(type_conversions[item])
		elif in_or_out == 'out':
			converted_types.append(POINTER(type_conversions[item]))

	if len(converted_types) == 1:
		converted_types = converted_types[0]
		
	return converted_types