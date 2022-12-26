# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 02:37:40 2018

@author: f
"""
import cupy as np

from layers.layer import Layer
from mode.Mode import Mode


class BatchNorm(Layer):

    def __init__(self, inp_shape, eps=1e-8, mode=Mode.TRAIN):
        assert type(inp_shape) == int or len(inp_shape) == 3, \
            " The batch norm layer currently works for only fully connected layers and convolution layers"
        self._eps = eps
        channel = inp_shape if type(inp_shape) == int else inp_shape[2]
        gamma = np.ones(channel, dtype=np.float32)
        beta = np.zeros(channel, dtype=np.float32)
        self._inpNorm = 0
        self._mean = 0
        self._var = 1
        self._running_mean = np.zeros(channel, dtype=np.float32)
        self._running_var = np.ones(channel, dtype=np.float32)
        super(BatchNorm, self).__init__(inp_shape, mode)
        self._params = {"gamma": gamma, "beta": beta}
        self._d_params = {"gamma": np.zeros_like(gamma, dtype=np.float32),
                          "beta": np.zeros_like(beta, dtype=np.float32)}
        self._trained = False

    def trainable_parameters(self):
        return list(self._params.keys())

    def forward_pass(self, inp):
        self._inp = inp
        if self._mode == Mode.TRAIN:
            axis = (0,) if len(self._inp.shape) == 2 else (0, 1, 2)
            self._mean = np.mean(self._inp, axis=axis, keepdims=True)
            self._var = np.var(self._inp, axis=axis, keepdims = True)
            self._inpNorm = (self._inp - self._mean) / np.sqrt(self._var + self._eps)
            self._running_mean = (0.9 * self._running_mean) + (0.1 * self._mean)
            self._running_var = (0.9 * self._running_var) + (0.1 * self._var)
            self._trained = True
        else:
            self._inpNorm = (self._inp - self._running_mean) / np.sqrt(self._running_var + self._eps)
        
        return (self._params["gamma"] * self._inpNorm) + self._params["beta"]

    def backward_pass(self, upstream):
        assert self._trained
        axis = (0,) if len(self._inp.shape) == 2 else (0,1,2)
        B = self._inp.shape[0] if len(self._inp.shape) == 2 else self._inp.shape[0] * self._inp.shape[1] * self._inp.shape[2]
        
        d_inp_norm = upstream * self._params["gamma"]
        inp_mean = self._inp - self._mean
        std_inv = 1. / np.sqrt(self._var + self._eps)
        d_var = np.sum(d_inp_norm * inp_mean, axis=axis) * -0.5 * std_inv ** 3
        d_mean = np.sum(d_inp_norm * -std_inv, axis=axis) + (d_var * -2. * np.mean(inp_mean, axis=axis))
        d_inp = (d_inp_norm * std_inv) + (d_var * 2 * inp_mean / B) + (d_mean / B)
        d_gamma = np.sum(upstream * self._inpNorm, axis=axis)
        d_beta = np.sum(upstream, axis=axis)

        self._inpNorm = 0
        self._mean = 0
        self._var = 1
        self._d_params["gamma"] = d_gamma
        self._d_params["beta"] = d_beta
        return d_inp

    def update_params(self, optimizer, step):
        for param in optimizer:
            self._params[param] += optimizer[param].calc_update(step, self._d_params[param])
