# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:54:20 2018

@author: f
"""
import numpy as np

from layers.layer import Layer
from mode.Mode import Mode


class Dropout(Layer):

    def __init__(self, inp_shape, p):
        super(Dropout, self).__init__(inp_shape)
        self.probability = p

    def forward_pass(self, inp):
        self._inp = inp
        # remember the star in front if inp.shape is to unpack the tuple
        if self.mode == Mode.TRAIN:
            return self._inp * ((np.random.rand(*self._inp.shape) < self.probability) / self.probability)
        else:
            return self._inp

    def backward_pass(self, upstream_grad):
        return upstream_grad
