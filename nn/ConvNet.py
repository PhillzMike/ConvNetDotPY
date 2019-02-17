# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:17:27 2018

@author: Fasipe Timilehin
"""
import copy
import math
import pickle
from builtins import range
from timeit import default_timer as timer

import numpy as np
from Mode import Mode


class Cnn:
    def __init__(self, feature_layers, classifier_layers, loss_function, optimizer, num_of_labels):
        #   First Convolution Layer
        # conv_p = 0.8
        # fstride_for_conv = 1
        # self.firstConv = Conv((32, 32, 3), (5, 5), 20, stride=fstride_for_conv)
        # self.firstRelu = Relu((28, 28, 20))
        # self.firstDropout = Dropout((28, 28, 20), conv_p)
        # self.firstMaxPool = MaxPool((28, 28, 20), 2, 2)
        #
        # #   Second Convolution Layer
        # sstride_for_conv = 1
        # self.secondConv = Conv((14, 14, 20), (3, 3), 50, stride=sstride_for_conv)
        # self.secondRelu = Relu((12, 12, 50))
        # self.secondDropout = Dropout((12, 12, 50), conv_p)
        # self.secondMaxPool = MaxPool((12, 12, 50), 2, 2)
        #
        # #   first fully connected layer
        # self.firstFc = FC(1800, 500)
        # self.firstFcRelu = Relu(500)
        #
        # #   Batch Normallization Layer
        # self.batchNorm = BatchNorm(500)
        #
        # #   Dropout layer
        # p = 0.6
        # self.dropout = Dropout(500, p)
        #
        # #   second fully connected layer
        # self.secondFc = FC(500, num_of_labels)

        # self.feature_extractor_layers = [self.firstConv, self.firstRelu, self.firstDropout, self.firstMaxPool,
        #                                  self.secondConv, self.secondRelu, self.secondDropout,  self.secondMaxPool]
        #
        # self.classifier_layers = [self.firstFc, self.batchNorm, self.firstFcRelu,
        #                           self.dropout, self.secondFc]
        self.feature_extractor_layers = feature_layers
        self.classifier_layers = classifier_layers

        self.layers = self.feature_extractor_layers + self.classifier_layers
        self.loss_function = loss_function
        # self.layers = [self.firstConv, self.firstMaxPool, self.secondConv, self.secondMaxPool, self.firstFc,
        #                self.dropout, self.secondFc]

        self._optimizer = optimizer
        self.optimizers = {layer: {param: copy.deepcopy(self._optimizer) for param in layer.trainable_parameters()}
                           for layer in self.layers if len(layer.trainable_parameters()) != 0}

    @staticmethod
    def next_batch(data, label, n):
        num = data.shape[0]
        arr = np.arange(num)
        np.random.shuffle(arr)
        for i in range(int(math.ceil(num / n))):
            yield data[arr[n * i:n * (i + 1)], :, :, :], label[arr[n * i:n * (i + 1)]]

    def test(self, images, valid_label):

        logits, fs_max_pool_shape = Cnn.__forward_pass(images, self.feature_extractor_layers,
                                                       self.classifier_layers, Mode.TEST)
        loss = self.loss_function.forward_pass(logits, valid_label)
        label = np.argmax(logits, axis=1)
        acc = np.mean(label == valid_label) * 100
        return loss, acc

    @staticmethod
    def __forward(inputs, cnn_layers, mode):
        inp = inputs
        for layer in cnn_layers:
            layer.mode = mode
            inp = layer.forward_pass(inp)
        return inp

    # TODO figure out a better way to get the shape of the last activation region, you get sha
    @staticmethod
    def __forward_pass(inputs, feature_extractor_layers, classifier_layers, mode):
        num_of_objects = inputs.shape[0]
        feature_extractor_output = Cnn.__forward(inputs, feature_extractor_layers, mode)
        feature_extractor_output_shape = feature_extractor_output.shape
        feature_extractor_output.shape = (num_of_objects, -1)
        classifier_scores = Cnn.__forward(feature_extractor_output, classifier_layers, mode)
        return classifier_scores, feature_extractor_output_shape

    @staticmethod
    def __backward(upstream_grad, cnn_layers):
        grad = upstream_grad
        for i in range(len(cnn_layers) - 1, -1, -1):
            layer = cnn_layers[i]
            grad = layer.backward_pass(grad)
        return grad

    @staticmethod
    def __backward_pass(d_logits, feature_extractor_layers, classifier_layers, feature_extractor_output_shape):
        classifier_grad_output = Cnn.__backward(d_logits, classifier_layers)
        classifier_grad_output.shape = feature_extractor_output_shape
        feature_extractor_grad_output = Cnn.__backward(classifier_grad_output, feature_extractor_layers)
        return feature_extractor_grad_output

    def train(self, data, label, valid_data, valid_label, batch, no_of_epochs):
        assert data.shape[0] == label.shape[0], " The traning data and training label must be the same number"
        assert valid_data.shape[0] == valid_label.shape[0], " The validation data and validation labe mustb be the" \
                                                            "same number"
        # hyperparameters
        step = 1e-3
        start = timer()
        loss = 0
        running_loss = 0
        for epoch in range(no_of_epochs):
            for X, Y in self.next_batch(data, label, batch):
                #   forward pass into the network
                logits, fs_max_pool_shape = Cnn.__forward_pass(X, self.feature_extractor_layers,
                                                               self.classifier_layers, Mode.TRAIN)
                #   calculate loss
                data_loss = self.loss_function.forward_pass(logits, Y)
                loss = data_loss

                running_loss += data_loss
                #   calculate loss gradient with respect to logits
                dscores = self.loss_function.backward_pass(Y)
                #   backward pass through the netowrk
                dx = Cnn.__backward_pass(dscores, self.feature_extractor_layers, self.classifier_layers,
                                         fs_max_pool_shape)
                del dx
                #   update parameters based on gradients calcuated
                for layer in self.layers:
                    if layer in self.optimizers:
                        layer.update_params(self.optimizers[layer], step)
            # calculate validation loss
            if (epoch + 1) % 1 == 0:
                valid_loss, acc = self.test(valid_data, valid_label)
                print("The validation loss is ", valid_loss, " Accuracy: ", acc)
                print("The loss after ", epoch, " iterations, learning rate is", step, "iterations is ",
                      running_loss / (data.shape[0] / batch), " using ",
                      timer() - start)

            running_loss = 0

    def save(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic
