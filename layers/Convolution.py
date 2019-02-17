# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:07:35 2018

@author: f
"""
import numpy as np

from layers.Pad import pad
from layers.layer import Layer


class Conv(Layer):

    def __init__(self, input_shape, filter_shape, no_of_filters, stride=1, padding=0):
        self._stride = stride
        self._pad = padding
        f = np.random.randn(*(filter_shape + (input_shape[2], no_of_filters))) / np.sqrt(
            (filter_shape[0] * filter_shape[1] * input_shape[2]) / 2)
        b = np.zeros(no_of_filters)

        super(Conv, self).__init__(input_shape)
        self._params = {"filter": f, "bias": b}
        self._d_params = {"filter": np.zeros_like(f), "bias": np.zeros_like(b)}

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
        self._d_params["bias"] = value

    def trainable_parameters(self):
        return list(self._params.keys())

    def forward_pass(self, inp):
        self._inp = inp
        filt = self._params["filter"]
        inp_trans = np.transpose(self._inp, (0, 3, 1, 2))
        filter_trans = np.transpose(filt, (3, 2, 0, 1))
        # TODO The transpose something
        w = inp_trans.shape[3]
        h = inp_trans.shape[2]
        n_filters, d_filter, h_filter, w_filter = filter_trans.shape
        w_out = int((w - w_filter + 2 * self._pad) / self._stride + 1)
        h_out = int((h - h_filter + 2 * self._pad) / self._stride + 1)
        x_col = self._im2col_indices(inp_trans, h_filter, w_filter, padding=self._pad, stride=self._stride)
        w_col = filter_trans.reshape(n_filters, -1)

        out = w_col @ x_col
        # Reshape back from 20x500 to 5x20x10x10
        # i.e. for each of our 5 images, we have 20 results with size of 10x10
        out = out.reshape(n_filters, h_out, w_out, inp_trans.shape[0])
        return out.transpose(3, 1, 2, 0) + self._params["bias"]

    def backward_pass(self, upstream_grad):
        filt = self._params["filter"]
        d_inp = np.zeros(self._inp.shape)
        df = np.zeros(filt.shape)
        act_reg = self._inp
        if self._pad > 0:
            # The pad method from Pad.py
            act_reg = pad(act_reg, self._pad)
            # db = np.zeros(b.shape)
        for h in range(upstream_grad.shape[1]):
            for w in range(upstream_grad.shape[2]):
                x = np.reshape(np.matmul(np.reshape(filt, (
                    filt.shape[0] ** 2 * filt.shape[2], filt.shape[3])),
                                         upstream_grad[:, h, w, :].T).T, (
                                   upstream_grad.shape[0], filt.shape[0], filt.shape[1],
                                   filt.shape[2]))
                d_inp[:, h * self._stride:h * self._stride + filt.shape[0],
                w * self._stride: w * self._stride + filt.shape[0], :] += x
                df += np.reshape(np.matmul(np.reshape(
                    act_reg[:, h * self._stride:h * self._stride + filt.shape[0],
                    w * self._stride: w * self._stride + filt.shape[0], :],
                    (act_reg.shape[0], filt.shape[0] ** 2 * act_reg.shape[3])).T, upstream_grad[:, h, w, :]),
                                 df.shape)

        db = np.sum(np.sum(upstream_grad, axis=(1, 2)), axis=0)
        self._d_params["filter"] = df
        self._d_params["bias"] = db
        return d_inp

    def update_params(self, optimizer, step):
        for param in optimizer:
            self._params[param] += optimizer[param].calc_update(step, self._d_params[param])

    # first implementation of the convolution layer, slow
#    def convolve(self,X,f,stride=1,pad=0):
#            n = X.shape[2]  #width of each image
#            h = X.shape[1] #hieght of each image
#            sizeOfFilter = f.shape[0]
#        
#            outWidth = (n-sizeOfFilter + 2 * pad)/stride + 1
#            outHieght = (h-sizeOfFilter + 2 * pad)/stride + 1
#                        
#            if(not outWidth.is_integer() or not outHieght.is_integer()):#checks if the output activation map is valid
#                raise ValueError("A valid output won't be produced")
#            
#            outWidth = int(outWidth)
#            outHieght = int(outHieght)
#            if(pad > 0):
#                X = self._pad(X,pad) 
#            numOfImages = X.shape[0]
#            sizeOfFilter = f.shape[0]
#            actReg = np.zeros((numOfImages,outHieght,outWidth))
#            startRow = 0
#            endRow = sizeOfFilter
#            
#            for i in range(outHieght):
#                startCol = 0
#                endCol = sizeOfFilter
#                for j in range(outWidth):
#                    actReg[:,i,j] = np.sum(np.multiply(X[:,startRow:endRow,startCol:endCol,:],f),axis = (1,2,3)) 
#                    startCol += stride
#                    endCol += stride
#                startRow += stride
#                endRow += stride
#            
#            return actReg
