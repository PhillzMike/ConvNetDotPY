# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:54:20 2018

@author: f
"""
import cupy as np

from layers.layer import Layer
from mode.Mode import Mode


class Dropout(Layer):

    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        assert 0 <= p < 1, "Dropout probability must be in range [0, 1)"
        self._p = p
        self._drop_prob = np.float32(1)

    def forward_pass(self, inp):
        if self.mode != Mode.TRAIN:
            return inp

        self._inp = inp
        self._drop_prob = ((np.random.rand(*self._inp.shape, dtype=np.float32) < (1 - self._p)) / (1 - self._p))
        # self._drop_prob = ((np.float32(np.random.rand(*self._inp.shape)) < self.probability) / self.probability)
        return self._inp * self._drop_prob

    def backward_pass(self, upstream_grad):
        grad = upstream_grad * self._drop_prob
        # del self._drop_prob
        return grad
