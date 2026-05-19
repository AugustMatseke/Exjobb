from abc import ABC, abstractmethod
import numpy as np


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
