import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def xor(np.ndarray[np.uint8_t, ndim=1] encode_model_, np.uint8_t byteskey):
    cdef np.ndarray[np.uint8_t, ndim=1] tmp = np.empty_like(encode_model_)
    for i in range(len(encode_model_)):
        tmp[i] = encode_model_[i] ^ byteskey
    return tmp