import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label


FOUR_NEIGHBOR_STRUCTURE = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=int)


def has_four_neighbor(grid, r, c):
    return (
        grid[r - 1, c] == 1
        or grid[r + 1, c] == 1
        or grid[r, c - 1] == 1
        or grid[r, c + 1] == 1
    )

def generate_tikz_contour(grid_size=10, hole_size=2):
    # 1. Initialize empty grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # 2. Create a random connected blob (Drunken Walk Expansion)
    center = grid_size // 2
    grid[center-2:center+2, center-2:center+2] = 1 # Seed core
    
    for _ in range(30): # Randomly expand edges
        edges = []
        for r in range(1, grid_size-1):
            for c in range(1, grid_size-1):
                if grid[r, c] == 0:
                    # 4-neighbor adjacency only (side-sharing, no corner-only links)
                    if has_four_neighbor(grid, r, c):
                        edges.append((r, c))
        if edges:
            idx = np.random.choice(len(edges))
            grid[edges[idx]] = 1

    # 3. Punch a hole
    # We find an "inner" point that is surrounded by 1s
    possible_holes = []
    for r in range(2, grid_size-2):
        for c in range(2, grid_size-2):
            if np.all(grid[r-1:r+2, c-1:c+2] == 1):
                possible_holes.append((r, c))
    
    if possible_holes:
        hr, hc = possible_holes[np.random.choice(len(possible_holes))]
        grid[hr:hr+hole_size, hc:hc+hole_size] = 0

    # Keep only the largest 4-connected component to avoid corner-connected outliers.
    labeled, num_labels = label(grid, structure=FOUR_NEIGHBOR_STRUCTURE)
    if num_labels > 1:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        largest_component = component_sizes.argmax()
        grid = (labeled == largest_component).astype(int)
    
    return grid

def extract_tikz_path(grid):
    """
    Rudimentary boundary tracer for orthogonal (Manhattan) paths.
    Returns a string formatted for TikZ: (x,y) -- (x,y) ...
    """
    # For a production tool, you'd use a proper wall-following algorithm.
    # Here we'll visualize the grid to let you pick coordinates or 
    # use this logic to see the 'occupied' blocks.
    rows, cols = grid.shape
    path_points = []
    
    fig, ax = plt.subplots()
    ax.imshow(grid, origin='lower', cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False) 
    ax.set_title("Random Partitioning Target")
    plt.show()

    print("\n--- TIKZ COORDINATE HINT ---")
    print("Cells marked 1 (Occupied):")
    for r in range(rows):
        for c in range(cols):
            if grid[r,c] == 1:
                print(f"({c},{r}) ", end="")
        print("")

# Run it
grid = generate_tikz_contour()
extract_tikz_path(grid)