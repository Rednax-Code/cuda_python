"""
This defines some data types which are handy for conversion.
"""

from ctypes import c_int, c_double, Structure, POINTER, memmove, cast, sizeof
from numpy.ctypeslib import as_array


class cArray(Structure):
	def __repr__(self):
		return str(as_array(self.data, shape=(self.length,)))
	
	def deep_copy(self):
		"""Creates a fast deep copy of the current cIntArray instance."""
		# Allocate new memory for the copied array
		new_data = (c_int * self.length)()  # Allocate memory directly in ctypes
		memmove(new_data, self.data, self.length * sizeof(c_int))  # Fast memory copy

		# Create a new cIntArray instance and assign the new pointer
		new_instance = cIntArray(as_array(self.data, shape=(self.length,)))
		new_instance.data = cast(new_data, POINTER(c_int))
		new_instance.length = self.length

		return new_instance
	
class cIntArray(cArray):
	_fields_ = [("data", POINTER(c_int)),
				("length", c_int)]
	
	def __init__(self, array):
		self.array = array
		ptr = array.ctypes.data_as(POINTER(c_int))
		super().__init__(ptr, len(array))
	
class cFloatArray(cArray):
	_fields_ = [("data", POINTER(c_double)),
				("length", c_int)]

	def __init__(self, array):
		self.array = array
		ptr = array.ctypes.data_as(POINTER(c_double))
		super().__init__(ptr, len(array))