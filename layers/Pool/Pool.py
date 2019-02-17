# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:11:56 2018

@author: f
"""
from abc import ABC

from layers.layer import Layer


class Pool(Layer, ABC):
    """An abstract class that models the pooling layers in a neural network"""

    def __init__(self, inp_shape, f, stride=1):
        self._filter = f
        self._stride = stride
        super(Pool, self).__init__(inp_shape)
        self._trained = False
