# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:07:04 2018

@author: f
"""
from optimizers.optimizer import Optimizer


class Momentum(Optimizer):

    def __init__(self, rho):
        self._velocity = 0
        self._rho = rho

    def calc_update(self, step, d_inp):
        self._velocity = (self._rho * self._velocity) + d_inp
        return -step * self._velocity
