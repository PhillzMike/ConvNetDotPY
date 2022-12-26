import numpy as np

from layers.activations.Activation import Activation


class Sigmoid(Activation):

    def __init__(self):
        super(Sigmoid, self).__init__()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, inp):
        self._inp = inp
        return Sigmoid.sigmoid(inp)

    def backward_pass(self, upstream_grad):
        sigmoid_inp = Sigmoid.sigmoid(self._inp)
        return upstream_grad * sigmoid_inp * (1 - sigmoid_inp)
