import numpy as np
from scipy.ndimage import label


FOUR_NEIGHBOR_STRUCTURE = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=int)


def _has_four_neighbor(grid, r, c):
    return (
        grid[r - 1, c] == 1
        or grid[r + 1, c] == 1
        or grid[r, c - 1] == 1
        or grid[r, c + 1] == 1
    )


def generate_grid(grid_size=20, hole_size=2) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    grid[center - 2:center + 2, center - 2:center + 2] = 1

    for _ in range(60):
        edges = [
            (r, c)
            for r in range(1, grid_size - 1)
            for c in range(1, grid_size - 1)
            if grid[r, c] == 0 and _has_four_neighbor(grid, r, c)
        ]
        if edges:
            grid[edges[np.random.choice(len(edges))]] = 1

    possible_holes = [
        (r, c)
        for r in range(2, grid_size - 2)
        for c in range(2, grid_size - 2)
        if np.all(grid[r - 1:r + 2, c - 1:c + 2] == 1)
    ]
    if possible_holes:
        hr, hc = possible_holes[np.random.choice(len(possible_holes))]
        grid[hr:hr + hole_size, hc:hc + hole_size] = 0

    labeled, num_labels = label(grid, structure=FOUR_NEIGHBOR_STRUCTURE)
    if num_labels > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        grid = (labeled == sizes.argmax()).astype(int)

    return grid
