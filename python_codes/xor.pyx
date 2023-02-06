
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.uint8_t[:] xor(np.uint8_t[:] encode_model_, np.uint8_t byteskey):
    cdef np.uint8_t[:] tmp = np.empty_like(encode_model_)
    cdef Py_ssize_t i
    for i in range(encode_model_.shape[0]):
        tmp[i] = encode_model_[i] ^ byteskey
    return tmp