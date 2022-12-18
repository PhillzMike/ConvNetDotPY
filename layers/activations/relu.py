# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:56:42 2018

@author: Fasipe Timilehin

"""
import cupy as np
from timeit import default_timer as timer

from layers.activations.Activation import Activation


class Relu(Activation):
    """This class performs the Rectified Linear Unit function used for non linearity in deep learning"""

    def __init__(self, inp_shape):
        super(Relu, self).__init__(inp_shape)

    def forward_pass(self, inp):
        # start = timer()
        self._inp = inp
        # result =  np.maximum(0, self._inp)
        # end = timer()

        # print("Forward pass - relu", end - start)
        return np.maximum(0, self._inp)

    def backward_pass(self, upstream):
        # start = timer()
        upstream[self._inp < 0] = 0
        # end = timer()
        # print("Backward pass - relu", end - start)
        return upstream
