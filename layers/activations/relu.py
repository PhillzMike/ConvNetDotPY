# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:56:42 2018

@author: Fasipe Timilehin

"""
import numpy as np

from layers.activations.Activation import Activation


class Relu(Activation):
    """This class performs the Rectified Linear Unit function used for non linearity in deep learning"""

    def __init__(self, inp_shape):
        super(Relu, self).__init__(inp_shape)

    def forward_pass(self, inp):
        self._inp = inp
        return np.maximum(0, self._inp)

    def backward_pass(self, upstream):
        upstream[self._inp < 0] = 0
        return upstream
