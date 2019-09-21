import numpy as np
cimport numpy as np

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def relu(np.ndarray[DTYPE_t, ndim=2] x):
    cdef np.ndarray[DTYPE_t, ndim=2] output = np.zeros(
        (x.shape[0], x.shape[1]),
        dtype=x.dtype)

    return output

cdef int inner_relu(np.ndarray[DTYPE_t, ndim=2] x,
                    np.ndarray[DTYPE_t, ndim=2] output, int number, int size) except? -1:
    for i in range(number):
        for j in range(size):
            if x[i, j] < 0:
                output[i, j] = 0
            else:
                output[i, j] = x[i, j]
