import numpy as np

from solver import Solver


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
            tl = int(grid[cy - 1, cx - 1] == 1) if cy > 0 and cx > 0 else 0
            tr = int(grid[cy - 1, cx]     == 1) if cy > 0 and cx < cols else 0
            bl = int(grid[cy,     cx - 1] == 1) if cy < rows and cx > 0 else 0
            br = int(grid[cy,     cx]     == 1) if cy < rows and cx < cols else 0

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

        candidates = []
        for rng, fn in [
            (range(cx - 1, -1, -1),     lambda n: h_chord(n, cx, cy)),
            (range(cx + 1, cols + 1),   lambda n: h_chord(cx, n, cy)),
            (range(cy - 1, -1, -1),     lambda n: v_chord(cx, n, cy)),
            (range(cy + 1, rows + 1),   lambda n: v_chord(cx, cy, n)),
        ]:
            for n in rng:
                chord = fn(n)
                if chord:
                    candidates.append(chord)
                    break

        if not candidates:
            return None
        return min(candidates, key=lambda d: abs(d[2] - d[0]) + abs(d[3] - d[1]))

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
                    if any(h_walls[next_r][cx] for cx in range(c0, c1 + 1)):
                        break
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
