import os

try:
    import _pycumath as _cumath
except ImportError:
    import pycumath._pycumath as _cumath

from pycumath.version import __version__
 
 
class PycuMath:
    def mul(input1, input2):
        _cumath.multiply_with_scalar(input1, input2)
        # return result
