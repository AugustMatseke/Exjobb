import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


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

def generate_tikz_contour(grid_size=20, hole_size=2):
    # 1. Initialize empty grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # 2. Create a random connected blob (Drunken Walk Expansion)
    center = grid_size // 2
    grid[center-2:center+2, center-2:center+2] = 1 # Seed core
    
    for _ in range(60): # Randomly expand edges
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

    print("\n--- TIKZ COORDINATE HINT ---")
    print("Cells marked 1 (Occupied):")
    for r in range(rows):
        for c in range(cols):
            if grid[r,c] == 1:
                print(f"({c},{r}) ", end="")
        print("")


def generate_single_width_candidates(grid):
    rows, cols = grid.shape
    candidates = []

    # Horizontal 1xk rectangles
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r, c] == 1:
                start = c
                while c < cols and grid[r, c] == 1:
                    c += 1
                end = c - 1
                for left in range(start, end + 1):
                    for right in range(left, end + 1):
                        cells = [(r, cc) for cc in range(left, right + 1)]
                        candidates.append({
                            "orientation": "H",
                            "r": r,
                            "c0": left,
                            "c1": right,
                            "cells": cells,
                        })
            else:
                c += 1

    # Vertical kx1 rectangles
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r, c] == 1:
                start = r
                while r < rows and grid[r, c] == 1:
                    r += 1
                end = r - 1
                for top in range(start, end + 1):
                    for bottom in range(top, end + 1):
                        cells = [(rr, c) for rr in range(top, bottom + 1)]
                        candidates.append({
                            "orientation": "V",
                            "c": c,
                            "r0": top,
                            "r1": bottom,
                            "cells": cells,
                        })
            else:
                r += 1

    return candidates


def minimum_single_width_rectangle_cover(grid):
    occupied_cells = [(r, c) for r, c in np.argwhere(grid == 1)]
    if not occupied_cells:
        return [], None

    candidates = generate_single_width_candidates(grid)
    num_cells = len(occupied_cells)
    num_candidates = len(candidates)

    cell_to_row = {cell: idx for idx, cell in enumerate(occupied_cells)}

    data = []
    row_idx = []
    col_idx = []
    for j, candidate in enumerate(candidates):
        for cell in candidate["cells"]:
            i = cell_to_row.get(cell)
            if i is not None:
                row_idx.append(i)
                col_idx.append(j)
                data.append(1.0)

    a = coo_matrix((data, (row_idx, col_idx)), shape=(num_cells, num_candidates))

    constraints = LinearConstraint(a, lb=np.ones(num_cells), ub=np.ones(num_cells))
    objective = np.ones(num_candidates)
    bounds = Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates))
    integrality = np.ones(num_candidates, dtype=int)

    result = milp(
        c=objective,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )

    if not result.success:
        raise RuntimeError(f"MILP failed to find a decomposition: {result.message}")

    selected_indices = np.where(result.x > 0.5)[0]
    return [candidates[idx] for idx in selected_indices], result


def draw_grid(ax, grid, title):
    rows, cols = grid.shape
    ax.imshow(grid, origin='lower', cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal', adjustable='box')

    # Explicitly draw a 1x1 white border around every cell.
    for r in range(rows):
        for c in range(cols):
            cell_border = plt.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                fill=False,
                edgecolor='white',
                linewidth=0.8,
                zorder=5,
            )
            ax.add_patch(cell_border)

    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.set_title(title)


def plot_rectangle_cover(ax, grid, rectangles):
    draw_grid(ax, grid, f"Minimum Non-Overlapping Single-Width Rectangles: {len(rectangles)}")

    cmap = plt.get_cmap("tab20")
    for i, rect in enumerate(rectangles):
        color = cmap(i % 20)
        for r, c in rect["cells"]:
            cell = plt.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor='white',
                linewidth=0.8,
                alpha=0.45,
            )
            ax.add_patch(cell)


def plot_side_by_side(grid, rectangles):
    _, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    draw_grid(axes[0], grid, "Random Partitioning Target")
    plot_rectangle_cover(axes[1], grid, rectangles)
    plt.show()

# Run it
grid = generate_tikz_contour()
extract_tikz_path(grid)
rectangles, milp_result = minimum_single_width_rectangle_cover(grid)
plot_side_by_side(grid, rectangles)

print(f"\nMinimum number of single-width non-overlapping rectangles: {len(rectangles)}")
if milp_result is not None:
    mip_gap = getattr(milp_result, "mip_gap", None)
    print(f"Solver status: {milp_result.status}")
    print(f"Solver message: {milp_result.message}")
    if mip_gap is not None:
        print(f"MILP relative gap: {mip_gap:.3e}")
        print("Global optimum certified:" + (" yes" if mip_gap <= 1e-9 else " no (gap > 0)"))

for idx, rect in enumerate(rectangles, start=1):
    if rect["orientation"] == "H":
        print(f"{idx:02d}. H: row={rect['r']}, cols={rect['c0']}..{rect['c1']}")
    else:
        print(f"{idx:02d}. V: col={rect['c']}, rows={rect['r0']}..{rect['r1']}")