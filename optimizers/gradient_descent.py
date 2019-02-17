# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:05:27 2018

@author: f
"""
from optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):

    def __init__(self):
        pass

    def calc_update(self, step, d_inp):
        return -step * d_inp
