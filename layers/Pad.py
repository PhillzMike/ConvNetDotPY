# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:45:06 2018

@author: f
"""
import numpy as np


def pad(X, pad, value=0):
    """

        :param self:
        :param X:
        :param pad:
        :param value:
        :return:
        """
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(value,))
