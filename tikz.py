import numpy as np


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

    if strips:
        lines.append("    % Strips")
        for s in strips:
            if s["orientation"] == "H":
                x0, y0, x1, y1 = s["c0"], s["r"], s["c1"] + 1, s["r"] + 1
            else:
                x0, y0, x1, y1 = s["c"], s["r0"], s["c"] + 1, s["r1"] + 1
            lines.append(
                f"    \\filldraw[fill=gray!20, draw=gray!60, line width=0.4pt]"
                f" ({x0},{y0}) rectangle ({x1},{y1});"
            )

    lines.append("    % Contour")
    for y, x1, x2 in h_segs:
        lines.append(f"    \\draw[black, very thick] ({x1},{y}) -- ({x2},{y});")
    for x, y1, y2 in v_segs:
        lines.append(f"    \\draw[black, very thick] ({x},{y1}) -- ({x},{y2});")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)
