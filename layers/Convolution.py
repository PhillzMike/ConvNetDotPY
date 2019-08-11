# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:07:35 2018

@author: f
"""

import numpy as np

from layers.im2col_cython import im2col_cython, col2im_cython
from layers.layer import Layer


class Conv(Layer):

    def __init__(self, input_shape, filter_shape, no_of_filters, stride=1, padding=0):
        self._stride = stride
        self._pad = padding
        f = np.random.randn(*(filter_shape + (input_shape[2], no_of_filters))) / np.sqrt(
            (filter_shape[0] * filter_shape[1] * input_shape[2]) / 2)
        b = np.zeros(no_of_filters)
        f = np.float32(f)
        b = np.float32(b)

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

    def forward_pass(self, inp):
        self._inp = inp
        inp_trans = np.transpose(self._inp, (0, 3, 1, 2))
        filter_trans = np.transpose(self.filter, (3, 2, 0, 1))

        w = inp_trans.shape[3]
        h = inp_trans.shape[2]
        n_filters, d_filter, h_filter, w_filter = filter_trans.shape
        w_out = int((w - w_filter + 2 * self._pad) / self._stride + 1)
        h_out = int((h - h_filter + 2 * self._pad) / self._stride + 1)

        x_col = im2col_cython(inp_trans, h_filter, w_filter, self._pad, self._stride)
        w_col = filter_trans.reshape(n_filters, -1)

        self._x_col = x_col
        out = self.mul(w_col, x_col)
        out = out.reshape(n_filters, h_out, w_out, inp_trans.shape[0])
        return out.transpose(3, 1, 2, 0) + self.bias

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
        dx = col2im_cython(dx_col, n, c, h, w, h_filter, w_filter, self._pad, self._stride)
        self._d_params["filter"] = dw
        self._d_params["bias"] = db
        return np.transpose(dx, (0, 2, 3, 1))

    def update_params(self, optimizer, step):
        for param in optimizer:
            self._params[param] += optimizer[param].calc_update(step, self._d_params[param])
