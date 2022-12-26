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

import cupy as np

from mode.Mode import Mode


class NN:
    def __init__(self, layers, loss_function, optimizer):
        self.layers = layers
        self.loss_function = loss_function
        self._optimizer = optimizer
        self.optimizers = {layer: {param: copy.deepcopy(self._optimizer) for param in layer.trainable_parameters()}
                           for layer in self.layers if len(layer.trainable_parameters()) != 0}

    @staticmethod
    def __next_batch(data, label, n):
        num = data.shape[0]
        arr = np.arange(num)
        np.random.shuffle(arr)
        for i in range(int(math.ceil(num / n))):
            yield data[arr[n * i:n * (i + 1)], :], label[arr[n * i:n * (i + 1)]]

    def get_logits(self, data):
        logits = NN.__forward_pass(data, self.layers, Mode.TEST)
        return logits

    def test(self, images):

        logits = self.get_logits(images)

        label = np.argmax(logits, axis=1)
        return logits, label
    
    def getValidationStats(self, images, labels, batch):
        if images.shape[0] == 0:
            return 0,0
        loss = 0
        accuracy = 0
        for image_b, label_b in NN.__next_batch(images, labels, batch):
            logits, label_gotten = self.test(image_b)
            loss += (self.loss_function.forward_pass(logits, label_b) * image_b.shape[0])
            accuracy += np.mean(label_gotten == label_b) * 100 * label_b.shape[0]
        return loss/images.shape[0], accuracy/images.shape[0]

    @staticmethod
    def __forward(inputs, cnn_layers, mode):
        inp = inputs
        for layer in cnn_layers:
            layer.mode = mode
            inp = layer.forward_pass(inp)
        return inp

    @staticmethod
    def __forward_pass(inputs, layers, mode):
        classifier_scores = NN.__forward(inputs, layers, mode)
        return classifier_scores

    @staticmethod
    def __backward(upstream_grad, cnn_layers):
        grad = upstream_grad
        for i in range(len(cnn_layers) - 1, -1, -1):
            layer = cnn_layers[i]
            grad = layer.backward_pass(grad)
        return grad

    @staticmethod
    def __backward_pass(d_logits, layers):
        feature_extractor_grad_output = NN.__backward(d_logits, layers)
        return feature_extractor_grad_output

    @staticmethod
    def __get_train_and_valid_data(data, label, validation_train_ratio):
        num = data.shape[0]
        arr = np.arange(num)
        np.random.shuffle(arr)

        train_length = int(len(data) * (1 - validation_train_ratio))
        return data[arr[:train_length], :], label[arr[:train_length]], data[arr[train_length:], :], label[
            arr[train_length:]]

    def train(self, data, label, validation_train_ratio, batch, no_of_epochs, learning_rate_func, print_every=1):
        assert data.shape[0] == label.shape[0], " The training data and training label must be the same number"
        assert 0 <= validation_train_ratio <= 1, "The validation to train ratio is wrong"

        train_data, train_label, valid_data, valid_label = NN.__get_train_and_valid_data(data, label,
                                                                                         validation_train_ratio)
        start = timer()
        training_loss = list()
        validation_loss = list()
        validation_accuracy = list()
        for epoch in range(1, no_of_epochs + 1):
            running_loss = 0
            learning_rate = learning_rate_func(epoch)
            for X, Y in NN.__next_batch(train_data, train_label, batch):
                #   forward pass into the network
                logits = NN.__forward_pass(X, self.layers, Mode.TRAIN)
                #   calculate loss
                data_loss = self.loss_function.forward_pass(logits, Y)
                running_loss += data_loss * X.shape[0]
                #   calculate loss gradient with respect to logits
                d_scores = self.loss_function.backward_pass(Y)
                #   backward pass through the network
                dx = NN.__backward_pass(d_scores, self.layers)
                del dx
                #   update parameters based on gradients calculated
                for layer in self.layers:
                    if layer in self.optimizers:
                        layer.update_params(self.optimizers[layer], learning_rate)
            np.cuda.get_current_stream().synchronize()
            training_loss.append(running_loss / train_data.shape[0])
            # calculate validation loss
            valid_loss, accuracy = self.getValidationStats(valid_data, valid_label, batch)
            np.cuda.get_current_stream().synchronize()
            validation_loss.append(valid_loss)
            validation_accuracy.append(accuracy)
            if epoch % print_every == 0:
                print("The validation loss is ", validation_loss[-1], " Accuracy: ", validation_accuracy[-1])
                print("The loss after ", epoch, " iterations, learning rate is", learning_rate, "iterations is ",
                      training_loss[-1], " using ", timer() - start)

        np.cuda.get_current_stream().synchronize()
        return training_loss, validation_loss, validation_accuracy

    def save(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic
