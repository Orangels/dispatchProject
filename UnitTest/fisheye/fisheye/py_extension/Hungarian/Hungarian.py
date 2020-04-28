import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer


class HungarianAlgorithm(object):
    def __init__(self):
        pwd = os.path.dirname(os.path.realpath(__file__)) + '/libHungarian.so'
        # '/home/user/project/run_retina/py_extension/Hungarian/libHungarian.so'
        self.lib = cdll.LoadLibrary(pwd)
        self.lib.init_Hungarian.restype = c_void_p
        self.obj = self.lib.init_Hungarian()

    def Solve(self, DistMatrix, row, col):
        DistMatrix_flatten = sum(DistMatrix, [])
        DistMatrix_flatten_num = row * col
        DistMatrix_flatten = (c_float * DistMatrix_flatten_num)(*DistMatrix_flatten)
        self.lib.Solve.argtypes = [c_void_p, (c_float * (DistMatrix_flatten_num)), c_int, c_int]
        self.lib.Solve.restype = ndpointer(dtype=c_int, shape=(col,))
        Solve_result = self.lib.Solve(self.obj, DistMatrix_flatten, row, col)
        Solve_result = Solve_result.astype(np.int32).tolist()
        return Solve_result
