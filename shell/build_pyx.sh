#!/usr/bin/env bash

# generate sample1.c
cython -X language_level=3 pack_vector_set.pyx

# generate sample1.so
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/usr/include/python3.6 \
    -I/opt/conda/lib/python3.6/site-packages/numpy/core/include \
    -o pack_vector_set.so pack_vector_set.c
