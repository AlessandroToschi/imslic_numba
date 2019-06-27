import numpy as np
import cv2
from numba import njit, prange
import time

sigma = 6.0 / 29.0
sigma_2 = np.power(sigma, 2.0)
sigma_3 = np.power(sigma, 3.0)
M = np.array([[0.4124564, 0.3575761, 0.1804375], 
              [0.2126729, 0.7151522, 0.0721750], 
              [0.0193339, 0.1191920, 0.9503041]])

@njit(cache=True)
def xyz_to_lab(f_xyz):
    lab = np.zeros((3))
    lab[0] = compute_l(f_xyz)
    lab[1] = compute_l(f_xyz)
    lab[2] = compute_l(f_xyz)
    return lab

@njit(cache=True)
def f(t):
    if t > sigma_3:
        return np.power(t, 1.0 / 3.0)
    else:
        return t / (3.0 * sigma_2) + (4.0 / 29.0)

@njit(cache=True)
def f_vector(xyz):
    f_xyz = np.zeros((3))
    f_xyz[0] = f(xyz[0])
    f_xyz[1] = f(xyz[1])
    f_xyz[2] = f(xyz[2])
    return f_xyz

@njit(cache=True)
def rgb_to_xyz(rgb):
    return M @ rgb

@njit(cache=True)
def compute_l(f_xyz):
    return 116.0 * f_xyz[1] - 16.0

@njit(cache=True)
def compute_a(f_xyz):
    return 500.0 * (f_xyz[0] / 0.950489 - f_xyz[1])

@njit(cache=True)
def compute_b(f_xyz):
    return 200.0 * (f_xyz[1] - f_xyz[2] / 1.08884)

@njit(cache=True)
def rgb_to_lab(image):
    scaled_image = image / 255.0
    lab_image = np.zeros_like(scaled_image)
    for y in range(scaled_image.shape[0]):
        for x in range(scaled_image.shape[1]):
            xyz = M @ scaled_image[y, x, :]
            lab_image[y, x, 0] = (116.0 * f(xyz[1]) - 16.0) * 2.55
            lab_image[y, x, 1] = 500.0 * (f(xyz[0]) / 0.950489 - f(xyz[1])) + 128.0
            lab_image[y, x, 2] = 200.0 * (f(xyz[1]) - f(xyz[2]) / 1.08884) + 128.0
    return lab_image

def main():
    source_image = cv2.imread("./1.jpg")
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    lab_image = rgb_to_lab(source_image.astype("float32"))
    #for channel in range(3):
    #    print(lab_image[:, :, channel].min(), lab_image[:, :, channel].max())

#forward_vector = np.vectorize(forward)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)