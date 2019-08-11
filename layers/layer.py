# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:12:26 2018

@author:
"""

from abc import ABC, abstractmethod

from mode.Mode import Mode


class Layer(ABC):

    # the input shape excluding the batch_size
    def __init__(self, inp_shape, mode=Mode.TRAIN):
        self.__inp_shape = inp_shape
        self.__inp = None
        self.mode = mode

    @property
    def _inp(self):
        return self.__inp

    @_inp.setter
    def _inp(self, value):
        assert value is not None, "Value cannot be None"
        # can be optimized, but it needs to be easy to understand
        if len(value.shape[1:]) != 1:
            if self.__inp_shape != value.shape[1:]:
                raise ValueError("value's shape ", value.shape, " does not correspond to the layer's shape ",
                                 self.__inp_shape)
        else:
            if self.__inp_shape != value.shape[1]:
                raise ValueError("value's shape ", value.shape, " does not correspond to the layer's shape ",
                                 self.__inp_shape)
        self.__inp = value

    @property
    def mode(self):
        return self._mode

    def mul(self, matrix_a, matrix_b):
        return matrix_a @ matrix_b

    @mode.setter
    def mode(self, value):
        if not isinstance(value, Mode):
            ValueError("value must be an enum Mode")

        self._mode = value

    def trainable_parameters(self):
        return []

    @abstractmethod
    def forward_pass(self, inp):
        pass

    @abstractmethod
    def backward_pass(self, upstream_grad):
        pass
