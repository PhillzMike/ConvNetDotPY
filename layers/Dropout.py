# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:54:20 2018

@author: f
"""
import cupy as np

from layers.layer import Layer
from mode.Mode import Mode


class Dropout(Layer):

    def __init__(self, inp_shape, p):
        super(Dropout, self).__init__(inp_shape)
        self.probability = np.float32(p)
        self._drop_prob = np.float32(1)

    def forward_pass(self, inp):
        self._inp = inp
        self._drop_prob = np.float32(1)
        if self.mode == Mode.TRAIN:
            self._drop_prob = ((np.random.rand(*self._inp.shape, dtype=np.float32) < self.probability) / self.probability)
            # self._drop_prob = ((np.float32(np.random.rand(*self._inp.shape)) < self.probability) / self.probability)
        result = self._inp * self._drop_prob
        return result

    def backward_pass(self, upstream_grad):
        grad = upstream_grad * self._drop_prob
        del self._drop_prob
        return grad
