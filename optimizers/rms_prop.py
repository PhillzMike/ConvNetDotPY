# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:08:26 2018

@author: f
"""

import numpy as np

from optimizers.optimizer import Optimizer


class RmsProp(Optimizer):

    def __init__(self, decay_rate, eps=1e-7):
        self.__decay_rate = decay_rate
        self.__eps = eps
        self.__grad_squared = 0

    def calc_update(self, step, d_inp):
        self.__grad_squared *= self.__decay_rate
        self.__grad_squared += (1 - self.__decay_rate) * d_inp * d_inp
        return -step * d_inp / (np.sqrt(self.__grad_squared) + self.__eps)
