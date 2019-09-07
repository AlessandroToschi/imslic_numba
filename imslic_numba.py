import numpy as np
import cv2
from numba import njit, prange, stencil
from numba import types
from numba.typed import Dict
import time
from rgb2lab import rgb_to_lab
from area import compute_area, compute_cumulative_area
from seeds import compute_seeds, unravel_index

@njit(cache=True)
def shortest_path(region_graph):
    for h in range(region_graph.shape[0]):
        for k in range(region_graph.shape[0]):
            if h != k:
                for s in range(region_graph.shape[0]):
                    if h != s:
                        if region_graph[h, k] + region_graph[k, s] < region_graph[h, s]:
                            region_graph[h, s] = region_graph[h, k] + region_graph[k, s]
                            region_graph[s, h] = region_graph[h, k] + region_graph[k, s]
    return region_graph

@njit(cache=True)
def compute_lambda_factor(seed_position, region_size, area, xi, height, width):
    x_min = int(max(0, seed_position[1] - region_size))
    x_max = int(min(width - 1, seed_position[1] + region_size))
    y_min = int(max(0, seed_position[0] - region_size)) 
    y_max = int(min(height - 1, seed_position[0] + region_size))
    sub_region_area = area[y_min : y_max + 1, x_min : x_max + 1].sum()
    return np.sqrt(xi / sub_region_area)

@njit(parallel=True)
def compute_regions_distances(K, seeds_positions, region_size, area, xi, height, width, lab_image, sp_dict):
    for k in prange(K):
        lambda_factor = compute_lambda_factor(seeds_positions[:, k], region_size, area, xi, height, width)
        offset = region_size * lambda_factor
        x_min = int(max(0, seeds_positions[1, k] - offset))
        x_max = int(min(width - 1, seeds_positions[1, k] + offset))
        y_min = int(max(0, seeds_positions[0, k] - offset))
        y_max = int(min(height - 1, seeds_positions[0, k] + offset))
        delta_x = (x_max - x_min) + 1
        delta_y = (y_max - y_min) + 1
        region_graph = np.zeros((delta_x * delta_y, delta_x * delta_y))
        for y in range(delta_y - 1):
            for x in range(delta_x):
                index = y * delta_x + x
                east_neighbor_index = index + 1
                south_neighbor_index = index + delta_x
                east_distance = np.sqrt(
                    1 + np.sum(np.power(lab_image[y_min + y, x_min + x, :] - lab_image[y_min + y, x_min + x + 1, :], 2.0))
                )
                south_distance = np.sqrt(
                    1 + np.sum(np.power(lab_image[y_min + y, x_min + x, :] - lab_image[y_min + y + 1, x_min + x, :], 2.0))
                )
                region_graph[index, east_neighbor_index] = east_distance
                region_graph[east_neighbor_index, index] = east_distance
                region_graph[index, south_neighbor_index] = south_distance
                region_graph[south_neighbor_index, index] = south_distance
        shortest_region_path = shortest_path(region_graph)
        paths = np.zeros((4, shortest_region_path.shape[0] - 1))
        seed_index = int((seeds_positions[0, k] - y_min) * delta_x + seeds_positions[1, k] - x_min)
        counter = 0
        for i in range(shortest_region_path.shape[0]):
            if i != seed_index:
                y_d, x_d = unravel_index(i, delta_x)
                y, x = y_d + y_min, x_d + x_min
                #paths[:, counter] = [y, x, shortest_region_path[i, seed_index], k]
                paths[0, counter] = y
                paths[1, counter] = x
                paths[2, counter] = shortest_region_path[i, seed_index]
                paths[3, counter] = k
                counter += 1
        sp_dict[k] = paths
    print("Fine iterazione")

def main():
    source_image = cv2.imread("./1.jpg")
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

    lab_image = rgb_to_lab(source_image.astype("float32"))

    padded_shape = source_image.shape + np.array([2, 2, 0])
    padded_image = np.zeros((padded_shape))
    padded_image[1:-1, 1:-1, :] = lab_image
    padded_image[1:-1, 0, :]    = padded_image[1:-1, 1, :]
    padded_image[1:-1, -1, :]   = padded_image[1:-1, -2, :]
    padded_image[0, 1:-1, :]    = padded_image[1, 1:-1, :]
    padded_image[-1, 1:-1, :]   = padded_image[-2, 1:-1, :]
    padded_image[0, 0, :]       = padded_image[1, 1, :]
    padded_image[0, -1, :]      = padded_image[1, -2, :]
    padded_image[-1, 0, :]      = padded_image[-2, 1, :]
    padded_image[-1, -1, :]     = padded_image[-2, -2, :]

    area = compute_area(padded_image)
    cumulative_area = compute_cumulative_area(area)
    area = area.reshape((lab_image.shape[0], lab_image.shape[1]))
    region_size = 10.0
    K = int((source_image.shape[0] * source_image.shape[1]) / (region_size ** 2.0))
    xi = cumulative_area[-1] * 4.0 / K
    seeds_positions, seeds_values = compute_seeds(lab_image, cumulative_area, K)
    max_iterations = 10

    for iteration in range(max_iterations):
        labels = np.ones_like(area) * -1.0
        global_distances = np.ones_like(area) * 1E12
        sp_dict = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
        compute_regions_distances(K, seeds_positions, region_size, area, xi, lab_image.shape[0], lab_image.shape[1], lab_image, sp_dict)
        break



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)