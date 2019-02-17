# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 02:37:40 2018

@author: f
"""
import numpy as np

from layers.layer import Layer
from mode.Mode import Mode


class BatchNorm(Layer):

    def __init__(self, inp_shape, eps=1e-8, mode=Mode.TRAIN):
        self._eps = eps
        gamma = np.ones(inp_shape)
        beta = np.zeros(inp_shape)
        self._inpNorm = 0
        self._mean = 0
        self._var = 1
        self._running_mean = 0
        self._running_var = 0
        super(BatchNorm, self).__init__(inp_shape, mode)
        self._params = {"gamma": gamma, "beta": beta}
        self._d_params = {"gamma": np.zeros_like(gamma), "beta": np.zeros_like(beta)}
        self._trained = False

    # @classmethod
    # def create_from_shapes(cls, inp_shape, eps, gamma, beta):
    #     gamma =
    #     beta =
    #     inp = np.zeros(inp_shape)
    #
    #     return cls(inp, eps, gamma, beta)

    def trainable_parameters(self):
        return list(self._params.keys())

    def forward_pass(self, inp):
        self._inp = inp
        if self._mode == Mode.TRAIN:
            self._mean = np.mean(self._inp, axis=0)
            self._var = np.var(self._inp, axis=0)
            self._inpNorm = (self._inp - self._mean) / np.sqrt(self._var + self._eps)
            self._running_mean = self._mean if not isinstance(self._running_mean, np.ndarray) \
                else (0.9 * self._running_mean) + (0.1 * self._mean)
            self._running_var = self._var if not isinstance(self._running_var, np.ndarray) \
                else (0.9 * self._running_var) + (0.1 * self._var)
            self._trained = True
        else:
            self._inpNorm = (self._inp - self._running_mean) / np.sqrt(self._running_var + self._eps)
        return (self._params["gamma"] * self._inpNorm) + self._params["beta"]

    def backward_pass(self, upstream):
        assert self._trained

        d_inp_norm = upstream * self._params["gamma"]
        inp_mean = self._inp - self._mean
        std_inv = 1. / np.sqrt(self._var + self._eps)
        d_var = np.sum(d_inp_norm * inp_mean, axis=0) * -0.5 * std_inv ** 3
        d_mean = np.sum(d_inp_norm * -std_inv, axis=0) + (d_var * -2. * np.mean(inp_mean, axis=0))
        d_inp = (d_inp_norm * std_inv) + (d_var * 2 * inp_mean / self._inp.shape[0]) + (d_mean / self._inp.shape[0])
        d_gamma = np.sum(upstream * self._inpNorm, axis=0)
        d_beta = np.sum(upstream, axis=0)

        self._inpNorm = 0
        self._mean = 0
        self._var = 1
        self._d_params["gamma"] = d_gamma
        self._d_params["beta"] = d_beta
        return d_inp

    def update_params(self, optimizer, step):
        for param in optimizer:
            self._params[param] += optimizer[param].calc_update(step, self._d_params[param])
