import numpy as np
from numba import njit

@njit(cache=True)
def compute_ne_corner(x, y, neighbors):
    average_lab =   (neighbors[1, -1, :] +
                    neighbors[1, 0, :] +
                    neighbors[0, -1, :] +
                    neighbors[0, 0, :]) / 4.0
    average_position = np.array([x, y]) + np.array([-0.5, 0.5])
    return np.hstack((average_position, average_lab))

@njit(cache=True)
def compute_nw_corner(x, y, neighbors):
    average_lab =   (neighbors[-1, -1, :] +
                    neighbors[-1, 0, :] +
                    neighbors[0, -1, :] +
                    neighbors[0, 0, :]) / 4.0
    average_position = np.array([x, y]) + np.array([-0.5, -0.5])
    return np.hstack((average_position, average_lab))

@njit(cache=True)
def compute_sw_corner(x, y, neighbors):
    average_lab =   (neighbors[-1, 1, :] +
                    neighbors[-1, 0, :] +
                    neighbors[0, 1, :] +
                    neighbors[0, 0, :]) / 4.0
    average_position = np.array([x, y]) + np.array([0.5, -0.5])
    return np.hstack((average_position, average_lab))

@njit(cache=True)
def compute_se_corner(x, y, neighbors):
    average_lab =   (neighbors[1, 1, :] +
                    neighbors[1, 0, :] +
                    neighbors[0, 1, :] +
                    neighbors[0, 0, :]) / 4.0
    average_position = np.array([x, y]) + 0.5
    return np.hstack((average_position, average_lab))

@njit(cache=True)
def compute_area(image):
    area = np.zeros((image.shape[0] - 2, image.shape[1] - 2))
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            neighbors = image[y - 1 : y + 2, x - 1 : x + 2, :]
            a1 = compute_nw_corner(x, y, neighbors)
            a2 = compute_sw_corner(x, y, neighbors)
            a3 = compute_se_corner(x, y, neighbors)
            a4 = compute_ne_corner(x, y, neighbors)
            a21 = a1 - a2
            a23 = a3 - a2
            a41 = a1 - a4
            a43 = a3 - a4
            norm_a21 = np.linalg.norm(a21)
            norm_a23 = np.linalg.norm(a23)
            norm_a41 = np.linalg.norm(a41)
            norm_a43 = np.linalg.norm(a43)
            area_123 = 0.5 * norm_a21 * norm_a23 * np.sqrt(1 - np.power(np.dot(a21, a23) / (norm_a21 * norm_a23), 2.0))
            area_341 = 0.5 * norm_a41 * norm_a43 * np.sqrt(1 - np.power(np.dot(a41, a43) / (norm_a41 * norm_a43), 2.0))
            area[y - 1, x - 1] = area_123 + area_341
    return area.flatten()

@njit(cache=True)
def compute_cumulative_area(area):
    cumulative_area = np.zeros_like(area)
    accumulator = area[0]
    cumulative_area[0] = accumulator
    for i in range(1, area.shape[0]):
        accumulator += area[i]
        cumulative_area[i] = accumulator
    return cumulative_area.flatten()