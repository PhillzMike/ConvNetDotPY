# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:06:57 2018

@author: f
"""

# from layers.layer import Layer
# import numpy as np
#
# class RnnLayer(Layer):
#
#     def __init__(self, inp, wxh, whh, prev_hidden_state):
#         self._wxh = wxh
#         self._whh = whh
#         self.prev_hidden_state = prev_hidden_state
#         super(RnnLayer,self).__init__(inp)
#
#     @classmethod
#     def create_from_shapes(cls,inp_size, hidden_size, output_size):
#         wxh = 0.01 * np.random.randn(hidden_size,inp_size.shape[0])
#         whh = 0.01 * np.random.randn(hidden_size, hidden_size)
#         prev_hidden_state = np.zeros()
#         inp = np.zeros(inp_size)
#         return cls(inp,wxh,whh)
