# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:23:20 2018

@author: f
"""

import numpy as np

from layers.Pool.Pool import Pool


class MaxPool(Pool):
    """A class that performs max pooling of an activation region in a neural network"""

    def __init__(self, inp_shape, filter_shape, stride=1):
        assert (inp_shape[0] - filter_shape) % stride == 0
        assert (inp_shape[1] - filter_shape) % stride == 0
        self._indices = 0
        super(MaxPool, self).__init__(inp_shape, filter_shape, stride)

    # def forward_pass(self, inp):
    #     self._inp = inp
    #     n, h, w, c = self._inp.shape
    #     inp_reshaped = np.transpose(self._inp, (0, 3, 1, 2))
    #     inp_reshaped = inp_reshaped.reshape(n*c, 1, h, w)
    #
    #     w_out = int((w - self._filter) / self._stride + 1)
    #     h_out = int((h - self._filter) / self._stride + 1)
    #
    #     inp_col = im2col_cython(inp_reshaped, self._filter, self._filter, 0, self._stride)
    #     self._inp_col_shape = inp_col.shape
    #     self._max_index = np.argmax(inp_col, axis=0)
    #     out = inp_col[self._max_index, range(self._max_index.size)]
    #     out = out.reshape(h_out, w_out, n, c)
    #     return out.transpose(2, 0, 1, 3)
    #
    # def backward_pass(self, upstream_grad):
    #
    #     n, h, w, c = self._inp.shape
    #     d_inp_col = np.empty(self._inp_col_shape, dtype=np.float32)
    #     upstream_grad_flattened = np.transpose(upstream_grad, (1, 2, 0, 3)).ravel()
    #
    #     d_inp_col[self._max_index, range(self._max_index.size)] = upstream_grad_flattened
    #     d_inp = col2im_cython(d_inp_col, n*c, 1, h, w, self._filter, self._filter, 0, self._stride)
    #     return np.transpose(d_inp.reshape(n, c, h, w), (0, 2, 3, 1))

    def forward_pass(self, inp):
        self._inp = inp
        n = self._inp.shape[2]  # width of each image
        h = self._inp.shape[1]  # height of each image
        f = self._filter
        stride = self._stride
        out_width = int((n - f) / stride + 1)
        out_height = int((h - f) / stride + 1)
        num_of_images = self._inp.shape[0]

        output = np.empty((num_of_images, out_height, out_width, self._inp.shape[3]), dtype=np.float32)
        start_row = 0
        end_row = f
        filter_depth = self._inp.shape[3]  # since the depth of the filter should be the same as the depth of the image
        indices = np.empty((out_height, out_width, filter_depth, 2, num_of_images), dtype=int)
        for i in range(out_height):
            start_col = 0
            end_col = f
            for j in range(out_width):
                output[:, i, j, :] = np.max(self._inp[:, start_row:end_row, start_col:end_col, :], axis=(1, 2))
                for k in range(filter_depth):
                    p = self._inp[:, start_row:end_row, start_col:end_col, k]
                    indices[i, j, k, :, :] = np.unravel_index(np.argmax(p.reshape(num_of_images, f * f), axis=1),
                                                              (f, f))
                start_col += stride
                end_col += stride
            start_row += stride
            end_row += stride
        self._indices = indices
        self._trained = True

        return output

    def backward_pass(self, upstream_grad):
        assert self._trained  # check to make sure forwardPass has been called

        stride = self._stride
        positional_index = self._indices
        d_inp = np.empty(self._inp.shape, dtype=np.float32)
        for i in range(upstream_grad.shape[1]):
            for j in range(upstream_grad.shape[2]):
                for k in range(upstream_grad.shape[3]):
                    d_inp[range(self._inp.shape[0]), positional_index[i, j, k, 0, :] + (i * stride),
                    positional_index[i, j, k, 1, :] + (j * stride), k:k + 1] = upstream_grad[:, i, j, k:k + 1]
        return d_inp
