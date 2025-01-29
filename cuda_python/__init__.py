"""
Cuda Python
-----------

This library can be used to easily load dlls containing cuda kernels and run them as if it were a pythonic function.
"""

from .base import load_kernel
from .var_converter import convert