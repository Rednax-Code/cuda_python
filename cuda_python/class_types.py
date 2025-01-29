"""
This defines some data types which are handy for conversion.
"""

from ctypes import c_int, c_double, POINTER, Structure
from numpy.ctypeslib import as_array

class cIntArray(Structure):
	_fields_ = [("data", POINTER(c_int)),
				("length", c_int)]
	def __repr__(self):
		return repr(as_array(self.data, shape=(self.length,)))
	
class cFloatArray(Structure):
	_fields_ = [("data", POINTER(c_double)),
				("length", c_double)]
	def __repr__(self):
		return repr(as_array(self.data, shape=(self.length,)))