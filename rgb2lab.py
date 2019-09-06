import numpy as np
from numba import njit, stencil

sigma = 6.0 / 29.0
sigma_2 = np.power(sigma, 2.0)
sigma_3 = np.power(sigma, 3.0)
M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])

@stencil()
def gamma(t):
    if t[0, 0, 0] <= 0.04045:
        return t[0, 0, 0] / 12.92
    else:
        return np.power((t[0, 0, 0] + 0.055) / 1.055, 2.4)

@njit(cache=True)
def f(t):
    if t > sigma_3:
        return np.power(t, 1.0 / 3.0)
    else:
        return t / (3.0 * sigma_2) + 2.0 * sigma / 3.0

@njit(cache=True)
def rgb_to_lab(image):
    scaled_image = image / 255.0
    scaled_image = gamma(scaled_image)
    lab_image = np.zeros_like(scaled_image)
    for y in range(scaled_image.shape[0]):
        for x in range(scaled_image.shape[1]):
            xyz = M @ scaled_image[y, x, :]
            lab_image[y, x, 0] = (116.0 * f(xyz[1]) - 16.0) * 2.55
            lab_image[y, x, 1] = 500.0 * (f(xyz[0] / 0.950456) - f(xyz[1])) + 128.0
            lab_image[y, x, 2] = 200.0 * (f(xyz[1]) - f(xyz[2] / 1.088754)) + 128.0
    return lab_image