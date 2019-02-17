# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:07:17 2018

@author: f
"""
import numpy as np

from optimizers.optimizer import Optimizer


class Adagrad(Optimizer):

    def __init__(self, eps=1e-7):
        self.__grad_square = 0
        self.__eps = eps

    def calc_update(self, step, d_inp):
        self.__grad_square += d_inp * d_inp
        return -step * d_inp / np.sqrt(self.__grad_square + self.__eps)
