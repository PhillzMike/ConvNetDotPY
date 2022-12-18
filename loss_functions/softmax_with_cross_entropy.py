import cupy as np
from timeit import default_timer as timer

class SoftmaxCrossEntropy:
    def __init__(self):
        self.prob = np.NaN
        self.__trained = False

    def forward_pass(self, logits, labels):
        # start = timer()
        assert logits.shape[0] == labels.shape[0], "logits and label must contain the same number of objects in a batch"
        batch_size = logits.shape[0]
        exp_logits = np.exp(logits)
        #   TODO figure a better way to do this guy
        self.prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss_for_each_image = -np.log(self.prob[range(batch_size), labels])
        loss = np.sum(loss_for_each_image) / batch_size
        #   free up space
        # print(loss_for_each_image)

        del exp_logits, loss_for_each_image
        self.__trained = True
        # end = timer()

        # print("Forward pass - softmax", end - start)

        # return np.float32(loss)
        return loss

    def backward_pass(self, labels):
        # start = timer()

        assert self.__trained, "backward pass cannot be called before forward pass"
        batch_size = labels.shape[0]
        d_logits = self.prob
        d_logits[range(batch_size), labels] -= 1
        d_logits /= batch_size
        # end = timer()
        # print("Backward pass - softmax", end - start)

        return d_logits
