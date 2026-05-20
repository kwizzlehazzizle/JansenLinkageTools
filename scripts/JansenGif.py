"""
Jansen Linkage Animation Generator

===================================
Simulates the 11-bar Jansen linkage (used in Strandbeest walking machines)
and generates a GIF animation of two crank revolutions with foot path tracing.

USAGE
-----
    # Default lengths:
        python scripts/JansenGif.py

    # Custom lengths via --Lengths (quoted for Windows cmd to avoid & expansion):
        python scripts/JansenGif.py --Lengths "a=38.0&b=41.5&c=39.3&d=40.1&e=55.8&f=39.4&g=36.7&h=65.7&i=49.0&j=50.0&k=61.9&l=7.8&m=15.0"

Bar lengths from diagram:
    a=38.0, b=41.5, c=39.3, d=40.1, e=55.8, f=39.4,
    g=36.7, h=65.7, i=49.0, j=50.0, k=61.9,
    l=7.8, m=15.0

Topology (from diagram):
    Fixed pivots: J0 (center, b/c/d junction) and J1 (offset by a horizontally, l vertically)
    Horizontal distance between fixed pivots = a = 38.0
    Vertical distance between fixed pivots = l = 7.8
    Crank (m) rotates about J1

    Joints:
        J0: Fixed pivot 1 (origin) — bars b, c, d
        J1: Fixed pivot 2 (a, -l) — crank m
        J2: Crank tip — bars m, j, k
        J3: Top vertex — bars e, b, j
        J4: Top-left vertex — bars e, d, f
        J5: Left-middle vertex — bars f, g, h
        J6: Center-bottom vertex — bars c, g, i, k
        J7: Foot — bars h, i

    Bars:
        b: J0↔J3,  c: J0↔J6,  d: J0↔J4
        e: J3↔J4,  f: J4↔J5,  g: J5↔J6
        h: J5↔J7,  i: J6↔J7,  j: J2↔J3
        k: J1↔J6,  m: J1↔J2 (crank)
"""

import argparse
import io
import os
import urllib.parse
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import imageio.v3 as imageio

# ── Parse arguments ──────────────────────────────────────────
DEFAULT_LENGTHS = {
    'a': 38.0, 'b': 41.5, 'c': 39.3, 'd': 40.1,
    'e': 55.8, 'f': 39.4, 'g': 36.7, 'h': 65.7,
    'i': 49.0, 'j': 50.0, 'k': 61.9, 'l': 7.8,
    'm': 15.0
}

parser = argparse.ArgumentParser(description='Generate Jansen linkage GIF animation')
parser.add_argument('--Lengths', type=str, default=None,
                    help='Query-string of bar lengths, e.g. a=38.0&b=41.5&...&m=15.0 (angle is ignored)')
args = parser.parse_args()

# Parse --Lengths: accept full URL, bare query string, or query after "?"
lengths_str = args.Lengths
if lengths_str:
    # Strip everything before the first "?" if present
    if '?' in lengths_str:
        lengths_str = lengths_str.split('?', 1)[1]
    # Strip any trailing fragment
    if '#' in lengths_str:
        lengths_str = lengths_str.rsplit('#', 1)[0]
    parsed = dict(urllib.parse.parse_qsl(lengths_str))
    user_lengths = {k: float(v) for k, v in parsed.items() if k in DEFAULT_LENGTHS}
    LENGTHS = {**DEFAULT_LENGTHS, **user_lengths}
else:
    LENGTHS = DEFAULT_LENGTHS.copy()

# ── Bar lengths ──────────────────────────────────────────────
print(f"Using lengths: {', '.join(f'{k}={v}' for k, v in sorted(LENGTHS.items()))}")

# Distance between fixed pivots (horizontal = a, vertical = l)
FIXED_DIST_X = LENGTHS['a']   # 38.0 (horizontal offset)
FIXED_DIST_Y = LENGTHS['l']   # 7.8 (vertical offset, J1 is lower)

# ── Solve linkage for a given crank angle ────────────────────
def solve_linkage(theta_deg, x0=None):
    """
    Given crank angle (degrees), solve for all joint positions.
    Returns array of 8 joint positions (each [x, y]).
    
    Args:
        theta_deg: Crank angle in degrees
        x0: Optional initial guess for [J3, J4, J5, J6, J7] positions
            Used for continuation from previous frame
    """
    theta = np.radians(theta_deg)

    # Fixed pivots
    J0 = np.array([0.0, 0.0])
    J1 = np.array([FIXED_DIST_X, FIXED_DIST_Y])

    # Crank tip (known from angle)
    J2 = J1 + LENGTHS['m'] * np.array([np.cos(theta), np.sin(theta)])

    # Unknowns: J3, J4, J5, J6, J7 (10 values)
    # Pack as [J3x, J3y, J4x, J4y, J5x, J5y, J6x, J6y, J7x, J7y]

    def constraints(x):
        J3 = np.array([x[0], x[1]])
        J4 = np.array([x[2], x[3]])
        J5 = np.array([x[4], x[5]])
        J6 = np.array([x[6], x[7]])
        J7 = np.array([x[8], x[9]])

        errs = []
        # b: J0↔J3
        errs.append(np.hypot(J3[0]-J0[0], J3[1]-J0[1]) - LENGTHS['b'])
        # j: J2↔J3
        errs.append(np.hypot(J3[0]-J2[0], J3[1]-J2[1]) - LENGTHS['j'])
        # e: J3↔J4
        errs.append(np.hypot(J4[0]-J3[0], J4[1]-J3[1]) - LENGTHS['e'])
        # d: J0↔J4
        errs.append(np.hypot(J4[0]-J0[0], J4[1]-J0[1]) - LENGTHS['d'])
        # f: J4↔J5
        errs.append(np.hypot(J5[0]-J4[0], J5[1]-J4[1]) - LENGTHS['f'])
        # c: J0↔J6
        errs.append(np.hypot(J6[0]-J0[0], J6[1]-J0[1]) - LENGTHS['c'])
        # k: J2↔J6 (connects crank tip to J6)
        errs.append(np.hypot(J6[0]-J2[0], J6[1]-J2[1]) - LENGTHS['k'])
        # g: J5↔J6
        errs.append(np.hypot(J6[0]-J5[0], J6[1]-J5[1]) - LENGTHS['g'])
        # h: J5↔J7
        errs.append(np.hypot(J7[0]-J5[0], J7[1]-J5[1]) - LENGTHS['h'])
        # i: J6↔J7
        errs.append(np.hypot(J7[0]-J6[0], J7[1]-J6[1]) - LENGTHS['i'])
        return errs

    # Use provided initial guess or default geometry-based guess
    if x0 is None:
        x0 = [
            LENGTHS['b'] * 0.3,  LENGTHS['b'] * 0.9,   # J3 guess
            -LENGTHS['d'] * 0.7, LENGTHS['d'] * 0.7,   # J4 guess
            -LENGTHS['f'] * 0.8, 0.0,                   # J5 guess
            LENGTHS['c'] * 0.5, -LENGTHS['c'] * 0.8,   # J6 guess
            -LENGTHS['h'] * 0.3, -LENGTHS['h'] * 0.9,  # J7 guess
        ]

    result = least_squares(constraints, x0, method='lm',
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)

    if not result.success:
        print(f"⚠ Warning: solver may not have fully converged at θ={theta_deg:.1f}° "
              f"(cost={result.cost:.2e})")

    J3 = np.array([result.x[0], result.x[1]])
    J4 = np.array([result.x[2], result.x[3]])
    J5 = np.array([result.x[4], result.x[5]])
    J6 = np.array([result.x[6], result.x[7]])
    J7 = np.array([result.x[8], result.x[9]])

    return np.array([J0, J1, J2, J3, J4, J5, J6, J7])


# ── Compute all frames with continuation ────────────────────
print("Computing linkage positions for all crank angles...")
NUM_FRAMES = 720  # 1 degree per frame × 2 revolutions = 720 frames
angles = np.linspace(0, 720, NUM_FRAMES, endpoint=False)

all_joints = []  # List of (8, 2) arrays
foot_path = []   # Accumulated foot trajectory

# Use continuation: previous frame's solution as initial guess for next
prev_x0 = None
for i, angle in enumerate(angles):
    joints = solve_linkage(angle, x0=prev_x0)
    all_joints.append(joints)
    foot_path.append(joints[7])  # J7 is the foot
    prev_x0 = list(joints[3:].flatten())  # [J3..J7] as flat list for next guess

    if (i + 1) % 120 == 0:
        print(f"  Solved {i+1}/{NUM_FRAMES} frames")

foot_path = np.array(foot_path)

# ── Animation ────────────────────────────────────────────────
print("Generating animation frames...")

# ── Bar colors (single source of truth) ─────────────────────
# Colors sorted highest→lowest hex value, assigned b→m alphabetically
BAR_COLORS = {
    'b': '#ff6b6b',   # coral (highest)
    'c': '#e8838b',   # pink
    'd': '#f39c12',   # amber
    'e': '#f1c40f',   # yellow
    'f': '#e74c3c',   # red
    'g': '#e67e22',   # orange
    'h': '#9b59b6',   # purple
    'i': '#3498db',   # blue
    'j': '#1abc9c',   # teal
    'k': '#2980b9',   # dark blue
    'm': '#1a5276',   # navy (lowest)
}

# Define bar connections for drawing
# Each tuple: (joint_index_a, joint_index_b, linewidth, bar_name, length)
# Color is looked up from BAR_COLORS
BARS = [
    (0, 3, 4.0, 'b', LENGTHS['b']),
    (0, 4, 4.0, 'd', LENGTHS['d']),
    (3, 4, 4.0, 'e', LENGTHS['e']),
    (0, 6, 4.0, 'c', LENGTHS['c']),
    (4, 5, 4.0, 'f', LENGTHS['f']),
    (5, 6, 4.0, 'g', LENGTHS['g']),
    (5, 7, 4.0, 'h', LENGTHS['h']),
    (6, 7, 4.0, 'i', LENGTHS['i']),
    (2, 6, 4.0, 'k', LENGTHS['k']),
    (2, 3, 4.0, 'j', LENGTHS['j']),
    (1, 2, 5.0, 'm', LENGTHS['m']),
]

# Compute axis limits
all_positions = np.array(all_joints).reshape(-1, 2)
x_margin = 15
y_margin = 15
xlim = [all_positions[:, 0].min() - x_margin, all_positions[:, 0].max() + x_margin]
ylim = [all_positions[:, 1].min() - y_margin, all_positions[:, 1].max() + y_margin]

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')

# Main axes for the linkage animation (right side, leaving room for table on left)
ax = fig.add_axes([0.22, 0.05, 0.75, 0.9])
ax.set_facecolor('#1a1a2e')
ax.set_aspect('equal')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.axis('off')

# Title
title_text = "Jansen Linkage — Strandbeest Walking Mechanism"
title = ax.text(0.5, 0.96, title_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=20, fontweight='bold',
                color='#ecf0f1', bbox=dict(boxstyle='round,pad=0.5',
                facecolor='#16213e', edgecolor='#0f3460', alpha=0.9))

# Angle label
angle_label = ax.text(0.98, 0.90, '', transform=ax.transAxes,
                      ha='right', va='top', fontsize=15, fontweight='bold', color='#f39c12')

# Foot path trace (will be updated each frame)
foot_line, = ax.plot([], [], color='#00ff88', linewidth=3.5, alpha=0.8)

# Bar lines
bar_lines = []
for j1, j2, lw, name, length in BARS:
    line, = ax.plot([], [], color=BAR_COLORS[name], linewidth=lw, solid_capstyle='round')
    bar_lines.append((line, j1, j2))

# ── Lengths table on the left side ───────────────────────────
table_ax = fig.add_axes([0.02, 0.12, 0.2, 0.75])
table_ax.axis('off')

# Table header
table_title = table_ax.text(0.5, 0.95, 'Bar Lengths', ha='center', va='top',
                            fontsize=20, fontweight='bold', color='#ecf0f1',
                            transform=table_ax.transAxes)

# Build table data (colors referenced from BAR_COLORS defined above)
bar_info = [
    ('Bar', 'Length'),
    ('─' * 12, '──────'),
    ('b', f"{LENGTHS['b']:.1f}"),
    ('c', f"{LENGTHS['c']:.1f}"),
    ('d', f"{LENGTHS['d']:.1f}"),
    ('e', f"{LENGTHS['e']:.1f}"),
    ('f', f"{LENGTHS['f']:.1f}"),
    ('g', f"{LENGTHS['g']:.1f}"),
    ('h', f"{LENGTHS['h']:.1f}"),
    ('i', f"{LENGTHS['i']:.1f}"),
    ('j', f"{LENGTHS['j']:.1f}"),
    ('k', f"{LENGTHS['k']:.1f}"),
    ('m', f"{LENGTHS['m']:.1f}"),
]

# Draw table background
table_bg = table_ax.add_patch(plt.Rectangle((0.05, 0.02), 0.9, 0.88,
    transform=table_ax.transAxes, facecolor='#16213e',
    edgecolor='#0f3460', alpha=0.9, zorder=0))

# Draw table rows
y_pos = 0.82
row_height = 0.055
for row_idx, (bar, length) in enumerate(bar_info):
    if row_idx == 0:  # Header row
        color = '#f39c12'
        weight = 'bold'
        size = 16
    elif row_idx == 1:  # Separator
        y_pos -= 0.01
        continue
    else:
        # Use the matching color from BAR_COLORS
        color = BAR_COLORS.get(bar, '#bdc3c7')
        weight = 'bold'
        size = 14

    table_ax.text(0.15, y_pos, bar, ha='left', va='center',
                  fontsize=size, fontweight=weight, color=color,
                  transform=table_ax.transAxes, family='monospace')
    table_ax.text(0.75, y_pos, length, ha='right', va='center',
                  fontsize=size, color=color,
                  transform=table_ax.transAxes, family='monospace')
    y_pos -= row_height

# Joint markers — scale proportionally with linkage span (matches web app visual consistency)
world_span = (xlim[1] - xlim[0]) + (ylim[1] - ylim[0])
base_radius = world_span * 0.002   # ~0.42 for default span (~210), grows for larger linkages

joint_circles = []
for i in range(8):
    circle = Circle((0, 0), base_radius, color='#f1c40f', ec='#e67e22', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    joint_circles.append(circle)

# Fixed pivot markers (larger, distinct)
for idx in [0, 1]:
    joint_circles[idx].set_radius(base_radius * 1.75)
    joint_circles[idx].set_facecolor('#e74c3c')
    joint_circles[idx].set_edgecolor('#c0392b')

# Crank circle (green, shows rotation)
crank_circle = Circle((0, 0), LENGTHS['m'], fill=False,
                       color='#2ecc71', linewidth=3, linestyle='--', alpha=0.5)
ax.add_patch(crank_circle)

frames_for_gif = []

for frame_idx in range(NUM_FRAMES):
    joints = all_joints[frame_idx]
    angle = angles[frame_idx]

    # Update bars
    for line, j1, j2 in bar_lines:
        pts = joints[[j1, j2]]
        line.set_data(pts[:, 0], pts[:, 1])

    # Update joint markers
    for idx, circle in enumerate(joint_circles):
        circle.center = joints[idx]

    # Update crank circle
    crank_circle.center = joints[1]  # J1 is crank pivot

    # Update foot path trace (accumulated)
    foot_line.set_data(foot_path[:frame_idx+1, 0], foot_path[:frame_idx+1, 1])

    # Update angle label (mod 360 so it displays 0–359 over both cycles)
    angle_label.set_text(f'Crank Angle: {angle % 360:.1f}°')

    # Capture frame using savefig to BytesIO (more reliable than buffer_rgba)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1,
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    frame_img = Image.open(buf)
    frame = np.array(frame_img)
    if frame.shape[2] == 4:  # RGBA -> RGB
        frame = frame[:, :, :3]
    frames_for_gif.append(frame)

    if frame_idx % 60 == 0:
        print(f"  Frame {frame_idx}/{NUM_FRAMES} — θ = {angle:.1f}°")

plt.close(fig)

# ── Save GIF ─────────────────────────────────────────────────
from datetime import datetime
from PIL import Image

output_dir = r"C:\Sources\JansenLinkage"
ts = datetime.now().strftime("%Y%m%d%H%M%S")
output_path = os.path.join(output_dir, f"jansen_linkage_{ts}.gif")
idx = 0
while os.path.exists(output_path):
    output_path = os.path.join(output_dir, f"jansen_linkage_{ts}_{idx:02d}.gif")
    idx += 1
print(f"\nSaving GIF to {output_path} ...")
print(f"  Total frames collected: {len(frames_for_gif)}")

# Verify frames are actually different
if len(frames_for_gif) >= 2:
    diff = np.abs(frames_for_gif[0].astype(int) - frames_for_gif[180].astype(int)).sum()
    print(f"  Pixel difference between frame 0 and 180: {diff}")
    diff2 = np.abs(frames_for_gif[0].astype(int) - frames_for_gif[1].astype(int)).sum()
    print(f"  Pixel difference between frame 0 and 1: {diff2}")

# Convert numpy arrays to PIL Images - use 'RGB' mode explicitly
pil_images = [Image.fromarray(frame, mode='RGB') for frame in frames_for_gif]

# Save without optimization to avoid frame collapsing
# Save first frame
pil_images[0].save(
    output_path,
    save_all=True,
    append_images=pil_images[1:],
    loop=0,
    duration=21,
    optimize=False  # Disable optimization to prevent frame collapsing
)

# Verify the saved GIF
verify_img = Image.open(output_path)
print(f"✅ Done! GIF saved: {output_path}")
print(f"   Frames in saved GIF: {verify_img.n_frames}")
print(f"   Duration: ~{verify_img.n_frames * 16 / 1000:.1f}s | Loop: infinite")
