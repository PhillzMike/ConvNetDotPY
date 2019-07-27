# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:56:42 2018

@author: Fasipe Timilehin

"""
from abc import ABC

from layers.layer import Layer


class Activation(Layer, ABC):
    """An abstract class that models the activations that would be used in the Library"""

    def __init__(self, inp_shape):
        super(Activation, self).__init__(inp_shape)
