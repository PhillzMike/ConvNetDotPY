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

    def __init__(self):
        super(Relu, self).__init__()

    def forward_pass(self, inp):
        self._inp = inp
        result =  np.maximum(0, self._inp)
        
        return result

    def backward_pass(self, upstream):
        upstream[self._inp < 0] = 0
        return upstream
