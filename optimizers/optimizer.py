# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:06:44 2018

@author: f
"""
from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def calc_update(self, step, d_inp):
        pass
