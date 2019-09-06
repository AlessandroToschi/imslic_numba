import numpy as np
from numba import njit

@njit(cache=True)
def unravel_index(index, width):
    y = index // width
    x = index % width
    return (y, x)

@njit(cache=True)
def compute_seeds(lab_image, cumulative_area, K):
    seeds_positions = np.zeros((2, K))
    seeds_values = np.zeros((3, K))
    start = 0
    for seed, area in enumerate(np.linspace(0, cumulative_area[-1], K)):
        for i in range(start, cumulative_area.shape[0]):
            if cumulative_area[i] >= area:
                y, x = unravel_index(i, lab_image.shape[1])
                seeds_positions[:, seed] = [y, x]
                seeds_values[:, seed] = lab_image[y, x, :]
                start = i + 1
                break
    return seeds_positions, seeds_positions