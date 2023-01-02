# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:08:10 2018

@author: f
"""
import cupy as np

from optimizers.optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, eps=1e-8, beta1=0.9, beta2=0.999):
        self.__eps = eps
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__first_moment = 0.0   
        self.__second_moment = 0.0
        self.__t = 1.0  # TODO write this guy a better way

    def calc_update(self, step, d_inp):
        self.__first_moment = self.__beta1 * self.__first_moment + ((1 - self.__beta1) * d_inp)
        # print("first moment")
        self.__second_moment = self.__beta2 * self.__second_moment + ((1 - self.__beta2) * d_inp * d_inp)
        first_unbias = self.__first_moment / (1 - (self.__beta1 ** self.__t))
        second_unbias = self.__second_moment / (1 - (self.__beta2 ** self.__t))
        self.__t += 1
        result = -step * first_unbias / (np.sqrt(second_unbias) + self.__eps)
        return result
