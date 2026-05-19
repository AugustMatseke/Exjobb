from grid import generate_grid
from milp_solver import MILPSolver
from rect_partition_solver import RectPartitionSolver
from tikz import grid_to_tikz
from viz import plot_all

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
