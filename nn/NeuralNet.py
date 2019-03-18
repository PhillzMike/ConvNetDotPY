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

from mode.Mode import Mode


class NN:
    def __init__(self, feature_layers, classifier_layers, loss_function, optimizer):
        self.feature_extractor_layers = feature_layers
        self.classifier_layers = classifier_layers

        self.layers = self.feature_extractor_layers + self.classifier_layers
        self.loss_function = loss_function

        self._optimizer = optimizer
        self.optimizers = {layer: {param: copy.deepcopy(self._optimizer) for param in layer.trainable_parameters()}
                           for layer in self.layers if len(layer.trainable_parameters()) != 0}

    @staticmethod
    def __next_batch(data, label, n):
        num = data.shape[0]
        arr = np.arange(num)
        np.random.shuffle(arr)
        print("data", data.shape)
        for i in range(int(math.ceil(num / n))):
            yield data[arr[n * i:n * (i + 1)], :], label[arr[n * i:n * (i + 1)]]

    def get_logits(self, data):
        logits, fs_max_pool_shape = NN.__forward_pass(data, self.feature_extractor_layers,
                                                      self.classifier_layers, Mode.TEST)
        return logits

    def test(self, images):

        logits = self.get_logits(images)

        label = np.argmax(logits, axis=1)
        return logits, label

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
        feature_extractor_output = NN.__forward(inputs, feature_extractor_layers, mode)
        feature_extractor_output_shape = feature_extractor_output.shape
        feature_extractor_output.shape = (num_of_objects, -1)
        classifier_scores = NN.__forward(feature_extractor_output, classifier_layers, mode)
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
        classifier_grad_output = NN.__backward(d_logits, classifier_layers)
        classifier_grad_output.shape = feature_extractor_output_shape
        feature_extractor_grad_output = NN.__backward(classifier_grad_output, feature_extractor_layers)
        return feature_extractor_grad_output

    def train(self, data, label, valid_data, valid_label, batch, no_of_epochs, learning_rate, print_every):
        assert data.shape[0] == label.shape[0], " The training data and training label must be the same number"
        assert valid_data.shape[0] == valid_label.shape[0], " The validation data and validation label must be the" \
                                                            "same number"
        # hyper-parameters
        start = timer()
        running_loss = 0
        training_loss = list()
        validation_loss = list()
        validation_accuracy = list()
        for epoch in range(1, no_of_epochs + 1):
            for X, Y in NN.__next_batch(data, label, batch):
                #   forward pass into the network
                logits, fs_max_pool_shape = NN.__forward_pass(X, self.feature_extractor_layers,
                                                              self.classifier_layers, Mode.TRAIN)
                #   calculate loss
                data_loss = self.loss_function.forward_pass(logits, Y)

                running_loss += data_loss
                #   calculate loss gradient with respect to logits
                d_scores = self.loss_function.backward_pass(Y)
                #   backward pass through the network
                dx = NN.__backward_pass(d_scores, self.feature_extractor_layers, self.classifier_layers,
                                        fs_max_pool_shape)
                del dx
                #   update parameters based on gradients calculated
                for layer in self.layers:
                    if layer in self.optimizers:
                        layer.update_params(self.optimizers[layer], learning_rate)
            training_loss.append(running_loss / (data.shape[0] / batch))
            # calculate validation loss
            logits, label_gotten = self.test(valid_data)
            valid_loss = self.loss_function.forward_pass(logits, valid_label)
            acc = np.mean(label_gotten == valid_label) * 100
            validation_loss.append(valid_loss)
            validation_accuracy.append(acc)
            if epoch % print_every == 0:
                print("The validation loss is ", valid_loss, " Accuracy: ", acc)
                print("The loss after ", epoch, " iterations, learning rate is", learning_rate, "iterations is ",
                      training_loss[-1], " using ", timer() - start)

            running_loss = 0
        return training_loss, validation_loss, validation_accuracy

    def save(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic
