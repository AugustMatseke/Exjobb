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
# Rectangle-partition solver
# ---------------------------------------------------------------------------

class RectPartitionSolver(Solver):
    """Minimum strip cover via rectangle partition using concave-vertex matching."""

    def solve(self, grid: np.ndarray) -> list[dict]:
        return self._rects_to_strips(self.solve_rects(grid))

    def solve_rects(self, grid: np.ndarray) -> list[tuple[int, int, int, int]]:
        concave = self._find_concave_vertices(grid)
        diagonals = self._find_good_diagonals(grid, concave)
        matching = self._solve_matching(concave, diagonals)
        chords = self._place_chords(grid, concave, diagonals, matching)
        return self._extract_rectangles(grid, chords)

    # ------------------------------------------------------------------
    # Step 1: concave vertices
    # ------------------------------------------------------------------

    def _find_concave_vertices(self, grid: np.ndarray) -> list[tuple[int, int]]:
        rows, cols = grid.shape
        concave = []
        for cy in range(rows + 1):
            for cx in range(cols + 1):
                tl = int(grid[cy - 1, cx - 1] == 1) if cy > 0 and cx > 0 else 0
                tr = int(grid[cy - 1, cx]     == 1) if cy > 0 and cx < cols else 0
                bl = int(grid[cy,     cx - 1] == 1) if cy < rows and cx > 0 else 0
                br = int(grid[cy,     cx]     == 1) if cy < rows and cx < cols else 0
                if tl + tr + bl + br == 3:
                    concave.append((cx, cy))
        return concave

    # ------------------------------------------------------------------
    # Step 2: good diagonals
    # ------------------------------------------------------------------

    def _find_good_diagonals(
        self, grid: np.ndarray, concave: list[tuple[int, int]]
    ) -> list[tuple[int, int, int, int]]:
        rows, cols = grid.shape
        vertex_set = set(concave)
        diagonals = []

        # Group by row (horizontal chords) and by column (vertical chords)
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for cx, cy in concave:
            by_row.setdefault(cy, []).append(cx)
            by_col.setdefault(cx, []).append(cy)

        for cy, xs in by_row.items():
            if cy == 0 or cy == rows:
                continue
            for i, cx1 in enumerate(xs):
                for cx2 in xs[i + 1:]:
                    a, b = min(cx1, cx2), max(cx1, cx2)
                    if all(
                        grid[cy - 1, cx] == 1 and grid[cy, cx] == 1
                        for cx in range(a, b)
                    ):
                        diagonals.append((a, cy, b, cy))

        for cx, ys in by_col.items():
            if cx == 0 or cx == cols:
                continue
            for i, cy1 in enumerate(ys):
                for cy2 in ys[i + 1:]:
                    a, b = min(cy1, cy2), max(cy1, cy2)
                    if all(
                        grid[cy, cx - 1] == 1 and grid[cy, cx] == 1
                        for cy in range(a, b)
                    ):
                        diagonals.append((cx, a, cx, b))

        return diagonals

    # ------------------------------------------------------------------
    # Step 3 & 4: adjacency matrix + maximum matching
    # ------------------------------------------------------------------

    def _solve_matching(
        self,
        concave: list[tuple[int, int]],
        diagonals: list[tuple[int, int, int, int]],
    ) -> dict[int, int]:
        idx = {v: i for i, v in enumerate(concave)}
        n = len(concave)
        adj: list[list[int]] = [[] for _ in range(n)]
        for cx1, cy1, cx2, cy2 in diagonals:
            u = idx.get((cx1, cy1))
            v = idx.get((cx2, cy2))
            if u is not None and v is not None:
                adj[u].append(v)
                adj[v].append(u)

        # Augmenting-path matching (general graph, simple O(VE))
        match = [-1] * n

        def try_augment(u: int, visited: list[bool]) -> bool:
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    if match[v] == -1 or try_augment(match[v], visited):
                        match[u] = v
                        match[v] = u
                        return True
            return False

        for u in range(n):
            if match[u] == -1:
                visited = [False] * n
                visited[u] = True
                try_augment(u, visited)

        return {u: match[u] for u in range(n) if match[u] != -1}

    # ------------------------------------------------------------------
    # Step 5: place chords
    # ------------------------------------------------------------------

    def _place_chords(
        self,
        grid: np.ndarray,
        concave: list[tuple[int, int]],
        diagonals: list[tuple[int, int, int, int]],
        matching: dict,
    ) -> list[tuple[int, int, int, int]]:
        rows, cols = grid.shape
        chords: list[tuple[int, int, int, int]] = []
        used = set()

        # Build a lookup: pair of vertex coords -> diagonal
        diag_lookup: dict[tuple, tuple] = {}
        for d in diagonals:
            cx1, cy1, cx2, cy2 = d
            diag_lookup[((cx1, cy1), (cx2, cy2))] = d
            diag_lookup[((cx2, cy2), (cx1, cy1))] = d

        # Matched pairs
        for u, v in matching.items():
            if u < v:
                pu, pv = concave[u], concave[v]
                d = diag_lookup.get((pu, pv)) or diag_lookup.get((pv, pu))
                if d:
                    chords.append(d)
                    used.add(u)
                    used.add(v)

        # Unmatched: extend to boundary
        for i, (cx, cy) in enumerate(concave):
            if i in used:
                continue
            # Determine which cell is empty (the reflex direction)
            tl = int(grid[cy - 1, cx - 1] == 1) if cy > 0 and cx > 0 else 0
            tr = int(grid[cy - 1, cx]     == 1) if cy > 0 and cx < cols else 0
            bl = int(grid[cy,     cx - 1] == 1) if cy < rows and cx > 0 else 0
            br = int(grid[cy,     cx]     == 1) if cy < rows and cx < cols else 0

            # Try horizontal extension first, then vertical
            chord = self._extend_to_boundary(grid, cx, cy, tl, tr, bl, br)
            if chord:
                chords.append(chord)

        return chords

    def _extend_to_boundary(
        self,
        grid: np.ndarray,
        cx: int, cy: int,
        tl: bool, tr: bool, bl: bool, br: bool,
    ) -> tuple[int, int, int, int] | None:
        rows, cols = grid.shape

        def h_chord(cx_start, cx_end, cy_fixed):
            a, b = min(cx_start, cx_end), max(cx_start, cx_end)
            if cy_fixed == 0 or cy_fixed == rows:
                return None
            if all(
                grid[cy_fixed - 1, x] == 1 and grid[cy_fixed, x] == 1
                for x in range(a, b)
            ):
                return (a, cy_fixed, b, cy_fixed)
            return None

        def v_chord(cx_fixed, cy_start, cy_end):
            a, b = min(cy_start, cy_end), max(cy_start, cy_end)
            if cx_fixed == 0 or cx_fixed == cols:
                return None
            if all(
                grid[y, cx_fixed - 1] == 1 and grid[y, cx_fixed] == 1
                for y in range(a, b)
            ):
                return (cx_fixed, a, cx_fixed, b)
            return None

        # Which directions are valid for this concave vertex
        # empty quadrant determines valid extension axes
        empty_tl = not tl
        empty_tr = not tr
        empty_bl = not bl
        empty_br = not br

        candidates = []

        # Horizontal: sweep left to find boundary, sweep right to find boundary
        for dx in range(1, cols + 1):
            nx = cx - dx
            if nx < 0:
                break
            c = h_chord(nx, cx, cy)
            if c:
                candidates.append(c)
                break
        for dx in range(1, cols + 1):
            nx = cx + dx
            if nx > cols:
                break
            c = h_chord(cx, nx, cy)
            if c:
                candidates.append(c)
                break

        # Vertical: sweep up/down
        for dy in range(1, rows + 1):
            ny = cy - dy
            if ny < 0:
                break
            c = v_chord(cx, ny, cy)
            if c:
                candidates.append(c)
                break
        for dy in range(1, rows + 1):
            ny = cy + dy
            if ny > rows:
                break
            c = v_chord(cx, cy, ny)
            if c:
                candidates.append(c)
                break

        if not candidates:
            return None
        # Pick shortest chord
        def length(d):
            return abs(d[2] - d[0]) + abs(d[3] - d[1])
        return min(candidates, key=length)

    # ------------------------------------------------------------------
    # Step 6: extract rectangles via flood-fill
    # ------------------------------------------------------------------

    def _extract_rectangles(
        self,
        grid: np.ndarray,
        chords: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        rows, cols = grid.shape

        # h_walls[cy][cx] = True means there is a cut on the top edge of cell (cy, cx)
        # i.e., between row cy-1 and row cy at column cx
        h_walls = [[False] * cols for _ in range(rows + 1)]
        v_walls = [[False] * (cols + 1) for _ in range(rows)]

        for cx1, cy1, cx2, cy2 in chords:
            if cy1 == cy2:  # horizontal chord at corner-row cy1
                cy = cy1
                for cx in range(cx1, cx2):
                    if 0 < cy < rows:
                        h_walls[cy][cx] = True
            else:           # vertical chord at corner-col cx1
                cx = cx1
                for cy in range(cy1, cy2):
                    if 0 < cx < cols:
                        v_walls[cy][cx] = True

        visited = np.zeros((rows, cols), dtype=bool)
        rects = []

        for r0 in range(rows):
            for c0 in range(cols):
                if grid[r0, c0] != 1 or visited[r0, c0]:
                    continue
                # Expand right
                c1 = c0
                while c1 + 1 < cols and grid[r0, c1 + 1] == 1 and not v_walls[r0][c1 + 1]:
                    c1 += 1
                # Expand down: all rows must have the same column span and no h-wall
                r1 = r0
                while r1 + 1 < rows:
                    next_r = r1 + 1
                    # Check h-wall on top of next row for all cols in [c0, c1]
                    if any(h_walls[next_r][cx] for cx in range(c0, c1 + 1)):
                        break
                    # Check all cells in next row exist and no v-wall cuts within span
                    if any(grid[next_r, cx] != 1 for cx in range(c0, c1 + 1)):
                        break
                    if any(v_walls[next_r][cx] for cx in range(c0 + 1, c1 + 1)):
                        break
                    r1 = next_r

                visited[r0:r1 + 1, c0:c1 + 1] = True
                rects.append((r0, r1, c0, c1))

        return rects

    # ------------------------------------------------------------------
    # Step 7: rectangles → strips
    # ------------------------------------------------------------------

    def _rects_to_strips(
        self, rects: list[tuple[int, int, int, int]]
    ) -> list[dict]:
        strips = []
        for rect_id, (r0, r1, c0, c1) in enumerate(rects):
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h <= w:
                for r in range(r0, r1 + 1):
                    strips.append({
                        "orientation": "H",
                        "r": r, "c0": c0, "c1": c1,
                        "cells": [(r, c) for c in range(c0, c1 + 1)],
                        "rect_id": rect_id,
                    })
            else:
                for c in range(c0, c1 + 1):
                    strips.append({
                        "orientation": "V",
                        "c": c, "r0": r0, "r1": r1,
                        "cells": [(r, c) for r in range(r0, r1 + 1)],
                        "rect_id": rect_id,
                    })
        return strips


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

    h_strips = [s for s in strips if s["orientation"] == "H"]
    v_strips = [s for s in strips if s["orientation"] == "V"]

    lines = ["\\begin{tikzpicture}[scale=0.6, line cap=round]"]

    if h_strips or v_strips:
        lines.append("    % Strips")
        for s in h_strips:
            r, c0, c1 = s["r"], s["c0"], s["c1"]
            lines.append(
                f"    \\filldraw[fill=gray!20, draw=gray!60, line width=0.4pt]"
                f" ({c0},{r}) rectangle ({c1 + 1},{r + 1});"
            )
        for s in v_strips:
            c, r0, r1 = s["c"], s["r0"], s["r1"]
            lines.append(
                f"    \\filldraw[fill=gray!20, draw=gray!60, line width=0.4pt]"
                f" ({c},{r0}) rectangle ({c + 1},{r1 + 1});"
            )

    lines.append("    % Contour")
    for y, x1, x2 in h_segs:
        lines.append(f"    \\draw[black, very thick] ({x1},{y}) -- ({x2},{y});")
    for x, y1, y2 in v_segs:
        lines.append(f"    \\draw[black, very thick] ({x},{y1}) -- ({x},{y2});")

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


def _color_strips(ax, strips):
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(strips):
        color = cmap(i % 20)
        for r, c in s["cells"]:
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=color, edgecolor='white', linewidth=0.8, alpha=0.45,
            ))


def _color_rects(ax, rects):
    cmap = plt.get_cmap("tab20")
    for i, (r0, r1, c0, c1) in enumerate(rects):
        color = cmap(i % 20)
        ax.add_patch(plt.Rectangle(
            (c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1,
            facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.45,
        ))


def plot_all(grid, milp_strips, rects, rect_strips):
    _, axes = plt.subplots(2, 2, figsize=(11.2, 11.2), constrained_layout=True)

    _draw_grid(axes[0, 0], grid, "Grid")
    _draw_grid(axes[0, 1], grid, f"MILP ({len(milp_strips)} strips)")
    _color_strips(axes[0, 1], milp_strips)
    _draw_grid(axes[1, 0], grid, f"Minimum rectangles ({len(rects)})")
    _color_rects(axes[1, 0], rects)
    _draw_grid(axes[1, 1], grid, f"Rect partition strips ({len(rect_strips)})")
    _color_strips(axes[1, 1], rect_strips)

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = generate_grid(hole_size=3)

    milp_strips = MILPSolver().solve(grid)
    rect_solver = RectPartitionSolver()
    rects = rect_solver.solve_rects(grid)
    rect_strips = rect_solver._rects_to_strips(rects)

    print(grid_to_tikz(grid, milp_strips))
    print(f"\nMILP:          {len(milp_strips)} strips")
    print(f"Rectangles:    {len(rects)}")
    print(f"RectPartition: {len(rect_strips)} strips")

    plot_all(grid, milp_strips, rects, rect_strips)
    