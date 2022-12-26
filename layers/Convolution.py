# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:07:35 2018

@author: f
"""

import math

import cupy as np
from timeit import default_timer as timer
import cupyx

from layers.im2col_cython import im2col_cython, col2im_cython
from layers.layer import Layer


class Conv(Layer):

    @staticmethod
    def calculate_fan_in_fan_out(input_dim, filter_shape, no_of_filters):
        fan_in = input_dim * filter_shape[0] * filter_shape[1]
        fan_out = no_of_filters * filter_shape[0] * filter_shape[1]

        return fan_in, fan_out

    def __init__(self, input_shape, filter_shape, no_of_filters, stride=1, padding=0):
        self._stride = stride
        self._pad = padding
        fan_in, fan_out = Conv.calculate_fan_in_fan_out(input_shape[-1], filter_shape, no_of_filters)
        gain = math.sqrt(2.0 / 6)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        f = np.random.uniform(-bound, bound, (filter_shape + (input_shape[2], no_of_filters)), dtype=np.float32)
        bound_for_bias = 1 / math.sqrt(fan_in)
        b = np.random.uniform(-bound_for_bias, bound_for_bias, no_of_filters, dtype=np.float32)
        # f = np.float32(f)
        # b = np.float32(b)

        super(Conv, self).__init__(input_shape)
        self._params = {"filter": f, "bias": b}
        self._d_params = {"filter": np.zeros_like(f, dtype=np.float32), "bias": np.zeros_like(b, np.float32)}

    @property
    def filter(self):
        return self._params["filter"]

    @filter.setter
    def filter(self, value):
        if self._params["filter"].shape != value.shape:
            raise ValueError("value shape does not correspond to the filter shape of the conv layer shape")
        self._params["filter"] = value

    @property
    def bias(self):
        return self._params["bias"]

    @bias.setter
    def bias(self, value):
        if self._params["bias"].shape != value.shape:
            raise ValueError("value shape does not correspond to the bias shape of the conv layer shape")
        self._params["bias"] = value

    def trainable_parameters(self):
        return list(self._params.keys())

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

    def forward_pass(self, inp):
        self._inp = inp
        inp_trans = np.transpose(self._inp, (0, 3, 1, 2))
        filter_trans = np.transpose(self.filter, (3, 2, 0, 1))

        w = inp_trans.shape[3]
        h = inp_trans.shape[2]
        n_filters, d_filter, h_filter, w_filter = filter_trans.shape
        w_out = int((w - w_filter + 2 * self._pad) / self._stride + 1)
        h_out = int((h - h_filter + 2 * self._pad) / self._stride + 1)

        x_col = self._im2col_indices(inp_trans, h_filter, w_filter, self._pad, self._stride)
        w_col = filter_trans.reshape(n_filters, -1)

        self._x_col = x_col
        out = self.mul(w_col, x_col)
        out = out.reshape(n_filters, h_out, w_out, inp_trans.shape[0])
        result = out.transpose(3, 1, 2, 0) + self.bias
        return result

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
        x = np.transpose(self._inp, (0, 3, 1, 2))
        w = np.transpose(self.filter, (3, 2, 0, 1))
        upstream_grad_trans = np.transpose(upstream_grad, (0, 3, 1, 2))
        n_filter, d_filter, h_filter, w_filter = w.shape

        db = np.sum(upstream_grad_trans, axis=(0, 2, 3))
        db = db.reshape(n_filter)

        d_upstream_reshaped = np.transpose(upstream_grad_trans, (1, 2, 3, 0)).reshape(n_filter, -1)
        dw = self.mul(d_upstream_reshaped, self._x_col.T)
        dw = dw.reshape(w.shape)
        dw = np.transpose(dw, (2, 3, 1, 0))

        w_reshape = w.reshape(n_filter, -1)
        dx_col = self.mul(w_reshape.T, d_upstream_reshaped)

        n, c, h, w = x.shape
        dx = self.col2im_indices(dx_col, (n, c, h, w), h_filter, w_filter, self._pad, self._stride)
        self._d_params["filter"] = dw
        self._d_params["bias"] = db

        return np.transpose(dx, (0, 2, 3, 1))

    def update_params(self, optimizer, step):
        for param in optimizer:
            result = optimizer[param].calc_update(step, self._d_params[param])
            self._params[param] += result
