"""
generate_rubik_icon.py
-----------------------
Draws a classic isometric Rubik's cube (3 visible faces, each a 3x3 grid of
colored tiles, mixed/scrambled) as a transparent-background PNG. Used as
the app's favicon and header logo -- see app.py.

Run: python assets/generate_rubik_icon.py
Output: assets/rubik_icon.png
"""

import math
import random
from PIL import Image, ImageDraw

# Classic Rubik's cube sticker colors
PALETTE = [
    (255, 255, 255),  # white
    (255, 213, 0),    # yellow
    (183, 18, 52),    # red
    (255, 88, 0),     # orange
    (0, 81, 186),     # blue
    (0, 158, 96),     # green
]

TILE = 60           # size of one sticker, in px
GAP_COLOR = (24, 28, 36, 255)   # dark cube-body/gap color
GAP = 4              # gap between stickers, in px

# Isometric basis vectors (unit length = TILE)
ANGLE = math.radians(30)
RIGHT = (math.cos(ANGLE) * TILE, math.sin(ANGLE) * TILE)
LEFT = (-math.cos(ANGLE) * TILE, math.sin(ANGLE) * TILE)
DOWN = (0.0, 1.0 * TILE)


def add(p, v, scale=1.0):
    return (p[0] + v[0] * scale, p[1] + v[1] * scale)


def shade(color, factor):
    return tuple(min(255, max(0, int(c * factor))) for c in color)


def draw_face(draw, origin, edge_a, edge_b, shade_factor, rng):
    """Draws a 3x3 grid of tiles spanning edge_a and edge_b from origin,
    over a dark full-face background so the gaps between tiles read as
    the cube's plastic body rather than transparent background."""
    face_corners = [
        origin,
        add(origin, edge_a, 3),
        add(add(origin, edge_a, 3), edge_b, 3),
        add(origin, edge_b, 3),
    ]
    draw.polygon(face_corners, fill=GAP_COLOR)

    for i in range(3):
        for j in range(3):
            c00 = add(add(origin, edge_a, i), edge_b, j)
            c10 = add(c00, edge_a, 1)
            c11 = add(add(c00, edge_a, 1), edge_b, 1)
            c01 = add(c00, edge_b, 1)
            # inset each tile slightly to render a visible gap/border
            cx = (c00[0] + c10[0] + c11[0] + c01[0]) / 4
            cy = (c00[1] + c10[1] + c11[1] + c01[1]) / 4
            inset_pts = []
            for p in (c00, c10, c11, c01):
                dx, dy = cx - p[0], cy - p[1]
                dist = math.hypot(dx, dy) or 1
                inset_pts.append((p[0] + dx / dist * GAP, p[1] + dy / dist * GAP))
            color = shade(rng.choice(PALETTE), shade_factor)
            draw.polygon(inset_pts, fill=color)


def generate(out_path="assets/rubik_icon.png", seed=7):
    rng = random.Random(seed)
    s = TILE

    # Compute canvas size from geometry, with padding
    width = int(6 * s * math.cos(ANGLE)) + 40
    height = int(3 * s + 3 * s * math.sin(ANGLE)) + 40

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    apex = (width / 2, 20)  # top vertex of the cube, roughly centered

    # Top face: spans RIGHT and LEFT from apex
    draw_face(draw, apex, RIGHT, LEFT, shade_factor=1.05, rng=rng)

    # Shared bottom vertex of the top rhombus
    bottom_of_top = add(add(apex, RIGHT, 3), LEFT, 3)
    left_vertex = add(apex, LEFT, 3)
    right_vertex = add(apex, RIGHT, 3)

    # Left face: spans RIGHT (to bottom_of_top) and DOWN, starting at left_vertex
    draw_face(draw, left_vertex, RIGHT, DOWN, shade_factor=0.72, rng=rng)

    # Right face: spans LEFT (to bottom_of_top) and DOWN, starting at right_vertex
    draw_face(draw, right_vertex, LEFT, DOWN, shade_factor=0.85, rng=rng)

    img.save(out_path)
    print(f"Wrote {out_path} ({width}x{height})")


if __name__ == "__main__":
    generate()
