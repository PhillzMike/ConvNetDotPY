from layers.activations.Activation import Activation


class LeakyRelu(Activation):

    def __init__(self, inp_shape):
        super(LeakyRelu, self).__init__(inp_shape)

    def forward_pass(self, inp):
        pass

    def backward_pass(self, upstream_grad):
        pass
