"""
generate_assets.py
-------------------
Generates the visual assets used on the app's Home tab: a hero banner and
a few minimalist icon graphics. Everything is drawn with matplotlib so the
app has zero dependency on external image hosting (nothing to break, no
licensing concerns).

Run once:
    python generate_assets.py
Produces (in ./assets/):
    hero_banner.png
    icon_growth.png
    icon_index.png
    icon_shield.png
    icon_bulb.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, Circle, RegularPolygon

os.makedirs("assets", exist_ok=True)

GREEN = "#0b3d2e"
GREEN2 = "#14532d"
SAFFRON = "#ff9933"
WHITE = "#ffffff"
RED = "#c0392b"


def hero_banner():
    fig, ax = plt.subplots(figsize=(15, 5), dpi=110)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Gradient background
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad = np.vstack([grad] * 2)
    ax.imshow(grad, extent=[0, 16, 0, 5], aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  "bg", [GREEN, GREEN2, SAFFRON]), zorder=0, alpha=0.95)

    # Candlesticks (stylised uptrend)
    rng = np.random.default_rng(11)
    n = 26
    xs = np.linspace(1, 15, n)
    base = np.linspace(1.2, 3.6, n) + rng.normal(0, 0.15, n)
    for i, x in enumerate(xs):
        o = base[i] + rng.uniform(-0.15, 0.15)
        c = base[i] + rng.uniform(-0.15, 0.15)
        h = max(o, c) + rng.uniform(0.05, 0.25)
        l = min(o, c) - rng.uniform(0.05, 0.25)
        color = WHITE if c >= o else "#2c2c2c"
        alpha = 0.9
        ax.plot([x, x], [l, h], color=color, linewidth=1.1, alpha=alpha, zorder=2)
        rect = mpatches.Rectangle((x - 0.18, min(o, c)), 0.36, max(abs(c - o), 0.05),
                                   facecolor=color, edgecolor=color, alpha=alpha, zorder=3)
        ax.add_patch(rect)

    # Smooth uptrend line
    ax.plot(xs, base + 0.35, color=WHITE, linewidth=2.5, alpha=0.85, zorder=4)

    plt.tight_layout(pad=0)
    plt.savefig("assets/hero_banner.png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()


def icon(filename, draw_fn, bg=WHITE):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    circle = Circle((5, 5), 4.6, facecolor=bg, edgecolor="none", zorder=0)
    ax.add_patch(circle)
    draw_fn(ax)
    plt.tight_layout(pad=0)
    plt.savefig(f"assets/{filename}", transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def draw_growth(ax):
    xs = [2.2, 3.6, 4.4, 5.6, 6.4, 7.8]
    ys = [3.2, 4.0, 3.7, 5.2, 4.9, 6.8]
    ax.plot(xs, ys, color=GREEN2, linewidth=2.8, solid_capstyle="round", zorder=2)
    arrow = FancyArrow(xs[-2], ys[-2], xs[-1] - xs[-2], ys[-1] - ys[-2],
                        width=0.03, head_width=0.55, head_length=0.55,
                        length_includes_head=True, color=GREEN2, zorder=3)
    ax.add_patch(arrow)
    for x, y in zip(xs[:-1], ys[:-1]):
        ax.add_patch(Circle((x, y), 0.14, color=SAFFRON, zorder=3))


def draw_index(ax):
    heights = [3.0, 4.6, 3.8, 6.2, 5.0]
    xs = [2.6, 3.9, 5.2, 6.5, 7.8]
    colors = [GREEN2, SAFFRON, GREEN2, SAFFRON, GREEN2]
    for x, h, c in zip(xs, heights, colors):
        ax.add_patch(mpatches.Rectangle((x - 0.45, 2), 0.9, h - 2, facecolor=c, edgecolor="none", zorder=2))


def draw_shield(ax):
    pts = np.array([
        [5, 8.3], [7.6, 7.1], [7.6, 4.6], [5, 1.9], [2.4, 4.6], [2.4, 7.1],
    ])
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=GREEN2, edgecolor="none", zorder=2))
    ax.plot([3.7, 4.6, 6.4], [5.1, 3.9, 6.3], color=WHITE, linewidth=1.8,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)


def draw_bulb(ax):
    ax.add_patch(Circle((5, 6), 2.1, facecolor=SAFFRON, edgecolor="none", zorder=2))
    ax.add_patch(mpatches.Rectangle((4.1, 3.2), 1.8, 1.2, facecolor="#555555", edgecolor="none", zorder=2))
    for i in range(3):
        ax.plot([4.3, 5.7], [3.2 + i * 0.35, 3.2 + i * 0.35], color=WHITE, linewidth=0.8, zorder=3)


if __name__ == "__main__":
    hero_banner()
    icon("icon_growth.png", draw_growth, bg="#e8f5e9")
    icon("icon_index.png", draw_index, bg="#fdf1e3")
    icon("icon_shield.png", draw_shield, bg="#e8f5e9")
    icon("icon_bulb.png", draw_bulb, bg="#fdf1e3")
    print("Saved assets to ./assets/")
