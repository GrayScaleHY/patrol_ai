# imreg.py

# Copyright (c) 2011-2022, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""FFT based image registration.
Imreg is a Python library that implements an FFT-based technique for
translation, rotation and scale-invariant image registration [1].
:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2022.9.27
Requirements
------------
* `CPython >= 3.7 <https://www.python.org>`_
* `Numpy 1.15 <https://www.numpy.org>`_
* `Scipy 1.5 <https://www.scipy.org>`_
* `Matplotlib 3.3 <https://www.matplotlib.org>`_  (optional for plotting)
Revisions
---------
2022.9.27
- Fix scipy.ndimage DeprecationWarning.
Notes
-----
Imreg is no longer being actively developed.
This implementation is mainly for educational purposes.
An improved version is being developed at https://github.com/matejak/imreg_dft.
References
----------
1. An FFT-based technique for translation, rotation and scale-invariant
   image registration. BS Reddy, BN Chatterji.
   IEEE Transactions on Image Processing, 5, 1266-1271, 1996
2. An IDL/ENVI implementation of the FFT-based algorithm for automatic
   image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
   Computers & Geosciences, 29, 1045-1055, 2003.
3. Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
   RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.
Examples
--------
>>> im0 = imread('t400')
>>> im1 = imread('Tr19s1.3')
>>> im2, scale, angle, (t0, t1) = similarity(im0, im1)
>>> imshow(im0, im1, im2)
>>> im0 = imread('t350380ori')
>>> im1 = imread('t350380shf')
>>> t0, t1 = translation(im0, im1)
>>> t0, t1
(20, 50)
"""

__version__ = '2022.9.27'

__all__ = (
    'translation',
    'similarity',
    'similarity_matrix',
    'logpolar',
    'highpass',
    'imread',
    'imshow',
)

import math

import numpy
import cv2

try:
    import cupy as xp
except ImportError:
    import numpy as xp

try:
    from cupy.fft import fft2, ifft2, fftshift
except:
    from numpy.fft import fft2, ifft2, fftshift

try:
    import cupyx.scipy.ndimage as ndimage
except ImportError:
    try:
        import scipy.ndimage as ndimage
    except ImportError:
        import ndimage  # type: ignore


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = xp.unravel_index(xp.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]


def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.
    A similarity transformation is an affine transformation with isotropic
    scale and without shear.
    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.
    """
    if im0.shape != im1.shape:
        raise ValueError('images must have same shapes')
    if len(im0.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = xp.unravel_index(xp.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base**i1

    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = xp.unravel_index(xp.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base**i1)
        if scale > 1.8:
            raise ValueError('images are not compatible. Scale change > 1.8')

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    center = im0.shape[1] // 2, im1.shape[0] // 2
    R = cv2.getRotationMatrix2D(center, float(angle), 1 / float(scale))
    im2 = cv2.warpAffine(im1 if isinstance(im1, numpy.ndarray) else im1.get(),
                         R, (im1.shape[1], im1.shape[0]))
    im2 = im2 if isinstance(im2, xp.ndarray) else xp.array(im2)
    f0 = fft2(im0)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = xp.unravel_index(xp.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    T = numpy.array([[1., 0., float(t1)], [0., 1., float(t0)]])
    im2 = cv2.warpAffine(im2 if isinstance(im2, numpy.ndarray) else im2.get(), 
                         T, (im2.shape[1], im2.shape[0]))
    im2 = im2 if isinstance(im2, xp.ndarray) else xp.array(im2)

    return im2, scale, angle, [-t0, -t1]


def similarity_matrix(scale, angle, vector):
    """Return homogeneous transformation matrix from similarity parameters.
    Transformation parameters are: isotropic scale factor, rotation angle
    (in degrees), and translation vector (of size 2).
    The order of transformations is: scale, rotate, translate.
    """
    S = xp.diag([scale, scale, 1.0])
    R = xp.identity(3)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[1, 1] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    T = xp.identity(3)
    T[:2, 2] = vector
    res = xp.array(T.get() @ R.get() @ S.get()) if xp.__name__ == "cupy" else T @ R @ S
    return res


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = xp.empty((angles, radii), dtype='float64')
    theta.T[:] = xp.linspace(0, xp.pi, angles, endpoint=False) * -1.0
    # d = radii
    d = xp.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = xp.empty_like(theta)
    radius[:] = (
        xp.power(log_base, xp.arange(radii, dtype='float64')) - 1.0
    )
    x = radius * xp.sin(theta) + center[0]
    y = radius * xp.cos(theta) + center[1]
    output = xp.empty_like(x)
    ndimage.map_coordinates(image, xp.array([x, y]), output=output)
    return output, log_base


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = xp.outer(
        xp.cos(xp.linspace(-math.pi / 2.0, math.pi / 2.0, shape[0])),
        xp.cos(xp.linspace(-math.pi / 2.0, math.pi / 2.0, shape[1])),
    )
    return (1.0 - x) * (2.0 - x)


def imread(fname, norm=True):
    """Return image data from img&hdr uint8 files."""
    with open(fname + '.hdr') as fh:
        hdr = fh.readlines()
    img = xp.fromfile(fname + '.img', xp.uint8, -1)
    img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
    if norm:
        img = img.astype('float64')
        img /= 255.0
    return img


def imshow(im0, im1, im2, im3=None, cmap=None, **kwargs):
    """Plot images using matplotlib."""
    from matplotlib import pyplot

    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.show()


if __name__ == '__main__':
    import os
    import doctest

    try:
        os.chdir('data')
    except Exception:
        pass
    doctest.testmod()