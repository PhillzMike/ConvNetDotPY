from layers.activations.Activation import Activation


class LeakyRelu(Activation):

    def __init__(self):
        super(LeakyRelu, self).__init__()

    def forward_pass(self, inp):
        pass

    def backward_pass(self, upstream_grad):
        pass
