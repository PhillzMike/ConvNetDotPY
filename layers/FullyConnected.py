# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:17:16 2018

@author: f
"""
import numpy as np

from layers.layer import Layer


class FC(Layer):

    def __init__(self, fan_in, fan_out):
        super(FC, self).__init__(fan_in)
        weight = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)
        weight = np.float32(weight)
        bias = np.zeros((1, fan_out))
        bias = np.float32(bias)
        self._params = {"weight": weight, "bias": bias}
        self._d_params = {"weight": np.zeros_like(weight), "bias": np.zeros_like(bias)}

    @property
    def weight(self):
        return self._params["weight"]

    @weight.setter
    def weight(self, value):
        if self._params["weight"].shape != value.shape:
            raise ValueError("The shape of the new weight does not correspond to the FC layer shape")
        self._params["weight"] = value

    @property
    def bias(self):
        return self._params["bias"]

    @bias.setter
    def bias(self, value):
        if self._params["bias"].shape != value.shape:
            raise ValueError("The shape of the new bias does not correspond to the FC layer shape")
        self._params["bias"] = value

    def trainable_parameters(self):
        return list(self._params.keys())

    def forward_pass(self, inp):
        self._inp = inp
        return np.dot(self._inp, self._params["weight"]) + self._params["bias"]

    def backward_pass(self, upstream_grad):
        dw = np.dot(self._inp.T, upstream_grad)
        db = np.sum(upstream_grad, axis=0, keepdims=True)
        d_inp = np.dot(upstream_grad, self.weight.T)

        self._d_params["weight"] = dw
        self._d_params["bias"] = db
        return d_inp

    def update_params(self, optimizer, step):
        for param in optimizer:
            self._params[param] += optimizer[param].calc_update(step, self._d_params[param])
