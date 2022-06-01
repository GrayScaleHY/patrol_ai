import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pack_vector_set(np.uint8_t[:, :] diff_image, np.int64_t[:] new):
    cdef Py_ssize_t i, j, i2, j2
    cdef np.uint8_t[:,:] block
    cdef np.uint8_t[25] feature
    cdef np.ndarray[np.uint8_t, ndim=2] feature_vector_set = \
        np.empty([(new[0] - 4) * (new[1] - 4), 25], dtype=np.uint8)

    cdef Py_ssize_t k = 0
    for i in range(2, new[0] - 2):
        for j in range(2, new[1] - 2):
            for i2 in range(5):
                for j2 in range(5):
                    feature_vector_set[k, i2 * 5 + j2] = diff_image[i - 2 + i2, j - 2 + j2]
            k += 1

    return feature_vector_set
