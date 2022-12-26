from layers.layer import Layer

class Flatten(Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward_pass(self, inp):
        self.__inp_shape = inp.shape
        result = inp.reshape(self.__inp_shape[0], -1)
        
        return result

    def backward_pass(self, upstream_grad):
        result = upstream_grad.reshape(*self.__inp_shape)
        
        return result
