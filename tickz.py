import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.ndimage import label
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


FOUR_NEIGHBOR_STRUCTURE = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=int)


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------

class Solver(ABC):
    """Base class for strip-cover solvers.

    Subclasses must implement solve(), returning a list of strip dicts with keys:
      - orientation: "H" or "V"
      - cells: list of (row, col) tuples covered by this strip
      - For "H": r, c0, c1
      - For "V": c, r0, r1
    """

    @abstractmethod
    def solve(self, grid: np.ndarray) -> list[dict]:
        ...


class MILPSolver(Solver):
    """Minimum single-width rectangle cover via Mixed Integer Linear Programming."""

    def solve(self, grid: np.ndarray) -> list[dict]:
        candidates = self._generate_candidates(grid)
        occupied_cells = [(r, c) for r, c in np.argwhere(grid == 1)]
        if not occupied_cells:
            return []

        num_cells = len(occupied_cells)
        num_candidates = len(candidates)
        cell_to_row = {cell: idx for idx, cell in enumerate(occupied_cells)}

        data, row_idx, col_idx = [], [], []
        for j, candidate in enumerate(candidates):
            for cell in candidate["cells"]:
                i = cell_to_row.get(cell)
                if i is not None:
                    row_idx.append(i)
                    col_idx.append(j)
                    data.append(1.0)

        a = coo_matrix((data, (row_idx, col_idx)), shape=(num_cells, num_candidates))
        constraints = LinearConstraint(a, lb=np.ones(num_cells), ub=np.ones(num_cells))
        bounds = Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates))
        integrality = np.ones(num_candidates, dtype=int)

        result = milp(
            c=np.ones(num_candidates),
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if not result.success:
            raise RuntimeError(f"MILP failed: {result.message}")

        selected = np.where(result.x > 0.5)[0]
        return [candidates[i] for i in selected]

    def _generate_candidates(self, grid: np.ndarray) -> list[dict]:
        rows, cols = grid.shape
        candidates = []

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
                            candidates.append({
                                "orientation": "H",
                                "r": r, "c0": left, "c1": right,
                                "cells": [(r, cc) for cc in range(left, right + 1)],
                            })
                else:
                    c += 1

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
                            candidates.append({
                                "orientation": "V",
                                "c": c, "r0": top, "r1": bottom,
                                "cells": [(rr, c) for rr in range(top, bottom + 1)],
                            })
                else:
                    r += 1

        return candidates


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TikZ output
# ---------------------------------------------------------------------------

def _merge_segments(segments):
    merged = []
    for key, start, end in sorted(segments):
        if not merged or merged[-1][0] != key or start > merged[-1][2]:
            merged.append([key, start, end])
        else:
            merged[-1][2] = max(merged[-1][2], end)
    return merged


def grid_to_tikz(grid: np.ndarray, strips: list[dict]) -> str:
    rows, cols = grid.shape

    h_segs, v_segs = [], []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 1:
                continue
            if r == 0 or grid[r - 1, c] == 0:
                h_segs.append((r, c, c + 1))
            if r == rows - 1 or grid[r + 1, c] == 0:
                h_segs.append((r + 1, c, c + 1))
            if c == 0 or grid[r, c - 1] == 0:
                v_segs.append((c, r, r + 1))
            if c == cols - 1 or grid[r, c + 1] == 0:
                v_segs.append((c + 1, r, r + 1))

    h_segs = _merge_segments(h_segs)
    v_segs = _merge_segments(v_segs)

    lines = ["\\begin{tikzpicture}[scale=0.6, line cap=round]"]

    lines.append("    % Contour")
    for y, x1, x2 in h_segs:
        lines.append(f"    \\draw[black, very thick] ({x1}, {y}) -- ({x2}, {y});")
    for x, y1, y2 in v_segs:
        lines.append(f"    \\draw[black, very thick] ({x}, {y1}) -- ({x}, {y2});")

    h_strips = [s for s in strips if s["orientation"] == "H"]
    v_strips = [s for s in strips if s["orientation"] == "V"]

    # if h_strips:
    #     lines.append("")
    #     lines.append("    % Horizontal strips")
    #     for s in h_strips:
    #         y = s["r"] + 0.5
    #         x1, x2 = s["c0"], s["c1"] + 1
    #         lines.append(f"    \\draw[black, thick] ({x1}, {y}) -- ({x2}, {y});")

    # if v_strips:
    #     lines.append("")
    #     lines.append("    % Vertical strips")
    #     for s in v_strips:
    #         x = s["c"] + 0.5
    #         y1, y2 = s["r0"], s["r1"] + 1
    #         lines.append(f"    \\draw[black, thick] ({x}, {y1}) -- ({x}, {y2});")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib visualisation
# ---------------------------------------------------------------------------

def _draw_grid(ax, grid, title):
    rows, cols = grid.shape
    ax.imshow(grid, origin='lower', cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor='white', linewidth=0.8, zorder=5,
            ))
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.set_title(title)


def plot_result(grid, strips):
    _, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    _draw_grid(axes[0], grid, "Grid")

    _draw_grid(axes[1], grid, f"Strip cover ({len(strips)} strips)")
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(strips):
        color = cmap(i % 20)
        for r, c in s["cells"]:
            axes[1].add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=color, edgecolor='white', linewidth=0.8, alpha=0.45,
            ))

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = generate_grid()
    solver = MILPSolver()
    strips = solver.solve(grid)

    print(grid_to_tikz(grid, strips))
    print(f"\n% {len(strips)} strips used")

    plot_result(grid, strips)
    