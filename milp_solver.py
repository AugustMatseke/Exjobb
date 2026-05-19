import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

from solver import Solver


class MILPSolver(Solver):
    """Minimum single-width rectangle cover via Mixed Integer Linear Programming."""

    def solve(self, grid: np.ndarray) -> list[dict]:
        candidates = self._generate_candidates(grid)
        occupied_cells = list(map(tuple, np.argwhere(grid == 1)))
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

        def scan(outer_size, inner_size, get, make):
            for i in range(outer_size):
                j = 0
                while j < inner_size:
                    if get(i, j) == 1:
                        start = j
                        while j < inner_size and get(i, j) == 1:
                            j += 1
                        for a in range(start, j):
                            for b in range(a, j):
                                candidates.append(make(i, a, b))
                    else:
                        j += 1

        scan(rows, cols,
             lambda r, c: grid[r, c],
             lambda r, a, b: {"orientation": "H", "r": r, "c0": a, "c1": b,
                              "cells": [(r, cc) for cc in range(a, b + 1)]})
        scan(cols, rows,
             lambda c, r: grid[r, c],
             lambda c, a, b: {"orientation": "V", "c": c, "r0": a, "r1": b,
                              "cells": [(rr, c) for rr in range(a, b + 1)]})

        return candidates
