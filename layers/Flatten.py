from layers.layer import Layer


class Flatten(Layer):

    def __init__(self, inp_shape):
        super(Flatten, self).__init__(inp_shape)

    def forward_pass(self, inp):
        self.__inp_shape = inp.shape
        return inp.reshape(self.__inp_shape[0], -1)

    def backward_pass(self, upstream_grad):
        return upstream_grad.reshape(*self.__inp_shape)
