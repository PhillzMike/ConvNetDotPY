import numpy as np

from layers.activations.Activation import Activation


class Tanh(Activation):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward_pass(self, inp):
        self._inp = inp
        return np.tanh(inp)

    def backward_pass(self, upstream_grad):
        return upstream_grad * (1 / np.power(np.cosh(self._inp), 2))
