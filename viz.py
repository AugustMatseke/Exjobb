import numpy as np
import matplotlib.pyplot as plt


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
