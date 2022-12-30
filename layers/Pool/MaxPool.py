# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:23:20 2018

@author: f
"""


import cupy as np
# import numpy as cp
import cupyx
from layers.im2col_cython import im2col_cython, col2im_cython

from layers.Pool.Pool import Pool


class MaxPool(Pool):
    """A class that performs max pooling of an activation region in a neural network"""

    def __init__(self, filter_shape, stride=1):
        self._indices = 0
        super(MaxPool, self).__init__(filter_shape, stride)

    def forward_pass(self, inp):
        self._inp = inp
        n, h, w, c = self._inp.shape
        inp_reshaped = np.transpose(self._inp, (0, 3, 1, 2))
        inp_reshaped = inp_reshaped.reshape(n * c, 1, h, w)

        w_out = int((w - self._filter) / self._stride + 1)
        h_out = int((h - self._filter) / self._stride + 1)

        inp_col = self._im2col_indices(inp_reshaped, self._filter, self._filter, 0, self._stride)
        self._inp_col_shape = inp_col.shape
        self._max_index = np.argmax(inp_col, axis=0)
        out = inp_col[self._max_index, range(self._max_index.size)]
        return out.reshape(h_out, w_out, n, c).transpose(2,0,1,3)
        

    def _get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_height) % stride == 0
        out_height = (H + 2 * padding - field_height) / stride + 1
        out_width = (W + 2 * padding - field_width) / stride + 1
        out_height = int(out_height)
        out_width = int(out_width)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    def _im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self._get_im2col_indices(x.shape, field_height, field_width, padding,
                                           stride)

        cols = x_padded[:, k, i, j]
        c = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * c, -1)
        return cols

    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self._get_im2col_indices(x_shape, field_height, field_width, padding,
                                           stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        # np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

    def backward_pass(self, upstream_grad):
        n, h, w, c = self._inp.shape
        d_inp_col = np.zeros(self._inp_col_shape, dtype=np.float32)
        upstream_grad_flattened = np.transpose(upstream_grad, (1, 2, 0, 3)).ravel()

        d_inp_col[self._max_index, range(self._max_index.size)] = upstream_grad_flattened
        d_inp = self.col2im_indices(d_inp_col, (n * c, 1, h, w), self._filter, self._filter, 0, self._stride)
        return d_inp.reshape(n, c, h, w).transpose(0,2,3,1)

    # def forward_pass(self, inp):
    #     self._inp = inp
    #     n = self._inp.shape[2]  # width of each image
    #     h = self._inp.shape[1]  # height of each image
    #     f = self._filter
    #     stride = self._stride
    #     out_width = int((n - f) / stride + 1)
    #     out_height = int((h - f) / stride + 1)
    #     num_of_images = self._inp.shape[0]
    
    #     output = np.empty((num_of_images, out_height, out_width, self._inp.shape[3]), dtype=np.float32)
    #     start_row = 0
    #     end_row = f
    #     filter_depth = self._inp.shape[3]  # since the depth of the filter should be the same as the depth of the image
    #     indices = np.empty((out_height, out_width, filter_depth, 2, num_of_images), dtype=np.int32)
    #     for i in range(out_height):
    #         start_col = 0
    #         end_col = f
    #         for j in range(out_width):
    #             output[:, i, j, :] = np.max(self._inp[:, start_row:end_row, start_col:end_col, :], axis=(1, 2))
    #             for k in range(filter_depth):
    #                 p = self._inp[:, start_row:end_row, start_col:end_col, k]
    #                 result = np.unravel_index(np.argmax(p.reshape(num_of_images, f * f), axis=1),
    #                                                           (f, f))
    #                 indices[i, j, k, 0, :] = result[0]
    #                 indices[i, j, k ,1, :] = result[1]
    #             start_col += stride
    #             end_col += stride
    #         start_row += stride
    #         end_row += stride
    #     self._indices = indices
    #     self._trained = True
    
    #     return output

    # def backward_pass(self, upstream_grad):
    #     assert self._trained  # check to make sure forwardPass has been called
    
    #     stride = self._stride
    #     positional_index = self._indices
    #     d_inp = np.empty(self._inp.shape, dtype=np.float32)
    #     for i in range(upstream_grad.shape[1]):
    #         for j in range(upstream_grad.shape[2]):
    #             for k in range(upstream_grad.shape[3]):
    #                 d_inp[range(self._inp.shape[0]), positional_index[i, j, k, 0, :] + (i * stride),
    #                 positional_index[i, j, k, 1, :] + (j * stride), k:k + 1] = upstream_grad[:, i, j, k:k + 1]
    #     return d_inp
