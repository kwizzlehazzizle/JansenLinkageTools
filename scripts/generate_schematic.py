"""
Jansen Linkage Minecraft Schematic Generator
=============================================
Generates a Create mod schematic JSON for a parameterized Jansen linkage.
Uses swivel bearings as joints and brass casings as bars.

Bar lengths can be provided via JSON config or use defaults from the
original linkage diagram (a=38.0, b=41.5, c=39.3, etc.).

The solver from jansen_linkage.py is reused to compute joint positions
at a given crank angle, then the JSON is built with correct bar lengths,
orientations, and swivel bearing connections.

Usage:
    python generate_schematic.py                          # default lengths
    python generate_schematic.py --config lengths.json    # custom lengths
    python generate_schematic.py --angle 90               # specific crank angle
"""

import json
import math
import struct
import hashlib
import argparse
import numpy as np
from scipy.optimize import least_squares

# ── Default bar lengths (from diagram) ──────────────────────
DEFAULT_LENGTHS = {
    'a': 38.0, 'b': 41.5, 'c': 39.3, 'd': 40.1,
    'e': 55.8, 'f': 39.4, 'g': 36.7, 'h': 65.7,
    'i': 49.0, 'j': 50.0, 'k': 61.9, 'l': 7.8,
    'm': 15.0
}

# Bar topology: each bar connects two joints (0-indexed)
BAR_CONNECTIONS = {
    'b': (0, 3), 'c': (0, 6), 'd': (0, 4),
    'e': (3, 4), 'f': (4, 5), 'g': (5, 6),
    'h': (5, 7), 'i': (6, 7), 'j': (2, 3),
    'k': (2, 6), 'm': (1, 2),
}

# Scale factor: linkage units -> Minecraft blocks
# Original schematic uses ~0.5 blocks per unit length
SCALE = 0.5

# Origin offset in Minecraft world (place the linkage here)
ORIGIN = {'x': 0.0, 'y': 20.0, 'z': 0.0}


# ── Linkage Solver (from jansen_linkage.py) ────────────────
def solve_linkage(theta_deg, lengths, x0=None):
    """
    Solve Jansen linkage for a given crank angle.
    Returns array of 8 joint positions [[x,y], [x,y], ...].
    """
    theta = math.radians(theta_deg)
    a = lengths['a']; l = lengths['l']; m = lengths['m']

    # Fixed pivots
    J0 = np.array([0.0, 0.0])
    J1 = np.array([a, l])

    # Crank tip
    J2 = J1 + m * np.array([math.cos(theta), math.sin(theta)])

    def constraints(x):
        J3 = np.array([x[0], x[1]])
        J4 = np.array([x[2], x[3]])
        J5 = np.array([x[4], x[5]])
        J6 = np.array([x[6], x[7]])
        J7 = np.array([x[8], x[9]])
        errs = []
        errs.append(np.hypot(J3[0]-J0[0], J3[1]-J0[1]) - lengths['b'])
        errs.append(np.hypot(J3[0]-J2[0], J3[1]-J2[1]) - lengths['j'])
        errs.append(np.hypot(J4[0]-J3[0], J4[1]-J3[1]) - lengths['e'])
        errs.append(np.hypot(J4[0]-J0[0], J4[1]-J0[1]) - lengths['d'])
        errs.append(np.hypot(J5[0]-J4[0], J5[1]-J4[1]) - lengths['f'])
        errs.append(np.hypot(J6[0]-J0[0], J6[1]-J0[1]) - lengths['c'])
        errs.append(np.hypot(J6[0]-J2[0], J6[1]-J2[1]) - lengths['k'])
        errs.append(np.hypot(J6[0]-J5[0], J6[1]-J5[1]) - lengths['g'])
        errs.append(np.hypot(J7[0]-J5[0], J7[1]-J5[1]) - lengths['h'])
        errs.append(np.hypot(J7[0]-J6[0], J7[1]-J6[1]) - lengths['i'])
        return errs

    if x0 is None:
        x0 = [
            lengths['b'] * 0.3,  lengths['b'] * 0.9,
            -lengths['d'] * 0.7, lengths['d'] * 0.7,
            -lengths['f'] * 0.8, 0.0,
            lengths['c'] * 0.5, -lengths['c'] * 0.8,
            -lengths['h'] * 0.3, -lengths['h'] * 0.9,
        ]

    result = least_squares(constraints, x0, method='lm',
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not result.success:
        print(f"  [warn] solver convergence at {theta_deg}°: cost={result.cost:.2e}")

    joints = np.vstack([J0, J1, J2,
                        result.x[0:2], result.x[2:4], result.x[4:6],
                        result.x[6:8], result.x[8:10]])
    return joints


# ── UUID Generation (deterministic from seed string) ───────
def make_uuid(seed):
    """Generate a deterministic 4-int UUID from a string seed."""
    h = hashlib.md5(seed.encode()).digest()
    return list(struct.unpack('iiii', h))


# ── Quaternion Math ────────────────────────────────────────
def quat_from_angle_z(angle_rad):
    """Quaternion for rotation around Z axis (maps local Y to world XY direction)."""
    half = angle_rad / 2
    return {
        'w': round(math.cos(half), 10),
        'x': 0.0,
        'y': 0.0,
        'z': round(math.sin(half), 10)
    }


# ── Bar Sub-Level Builder ──────────────────────────────────
def build_bar_sublevel(bar_name, joint_a, joint_b, lengths):
    """
    Build a sub_level dict for one bar.
    
    The bar extends along local Y axis. Local Y is rotated to align
    with the world-space direction from joint_a to joint_b.
    
    Returns (sub_level_dict, world_pos_of_joint_a_end, world_pos_of_joint_b_end).
    """
    # Direction and length in world space
    dx = joint_b[0] - joint_a[0]
    dy = joint_b[1] - joint_a[1]
    world_length = math.hypot(dx, dy)
    
    # Angle from +X axis toward the bar direction (for Z-axis rotation)
    # We want local +Y to point from joint_a to joint_b
    # Local +Y rotated by angle around Z should point in (dx, dy) direction
    # Y = (0,1) rotated by phi around Z gives (-sin(phi), cos(phi))
    # We want (-sin(phi), cos(phi)) = (dx/L, dy/L)
    # => sin(phi) = -dx/L, cos(phi) = dy/L
    # => phi = atan2(-dx, dy)
    angle = math.atan2(-dx, dy)
    
    # Number of brass casing blocks = round(length * scale)
    num_blocks = max(1, round(lengths[bar_name] * SCALE))
    
    # Size: [1, num_blocks+1, 2] (Y includes bearings at both ends)
    size = [1, num_blocks + 1, 2]
    
    # Blocks: brass casings along Y, swivel bearings at ends
    blocks = []
    # Brass casings (state 0 = create:brass_casing)
    for y in range(1, num_blocks):
        blocks.append({'pos': [0, y, 1], 'state': 0})
    
    # Swivel bearing at joint_a end (local pos [0, 0, 0]) - state 1
    blocks.append({
        'pos': [0, 0, 0], 'state': 1,
        'nbt': {
            'Speed': 0.0, 'NeedsSpeedUpdate': 1,
            'TargetAngle': 0.0,
            'SwivelCog': {'Speed': 0.0, 'NeedsSpeedUpdate': 1},
            'ScrollValue': 3,
            'id': 'simulated:swivel_bearing',
            # SubLevelID and SwivelPlate will be filled post-solve
        }
    })
    
    # Swivel bearing at joint_b end (local pos [0, num_blocks, 1]) - state 2
    blocks.append({
        'pos': [0, num_blocks, 1], 'state': 2,
        'nbt': {
            'Speed': 0.0, 'NeedsSpeedUpdate': 1,
            'TargetAngle': 0.0,
            'SwivelCog': {'Speed': 0.0, 'NeedsSpeedUpdate': 1},
            'ScrollValue': 3,
            'id': 'simulated:swivel_bearing',
        }
    })
    
    # World position: center of the bar (midpoint of joints, scaled)
    mid_x = (joint_a[0] + joint_b[0]) / 2 * SCALE + ORIGIN['x']
    mid_y = (joint_a[1] + joint_b[1]) / 2 * SCALE + ORIGIN['y']
    mid_z = ORIGIN['z']
    
    sub_level = {
        'orientation': quat_from_angle_z(angle),
        'size': size,
        'entities': [],
        'blocks': blocks,
        'palette': [
            {'Name': 'create:brass_casing'},
            {
                'Name': 'simulated:swivel_bearing',
                'Properties': {'powered': 'false', 'facing': 'north', 'assembled': 'true'}
            },
            {
                'Name': 'simulated:swivel_bearing',
                'Properties': {'powered': 'false', 'facing': 'south', 'assembled': 'true'}
            }
        ],
        'DataVersion': 3955,
        'position': {
            'x': round(mid_x, 8),
            'y': round(mid_y, 8),
            'z': round(mid_z, 8)
        },
        'uuid': make_uuid(f'bar_{bar_name}'),
        '_meta': {
            'bar_name': bar_name,
            'joint_a_idx': BAR_CONNECTIONS[bar_name][0],
            'joint_b_idx': BAR_CONNECTIONS[bar_name][1],
            'bearing_a_local': [0, 0, 0],
            'bearing_b_local': [0, num_blocks, 1],
        }
    }
    
    return sub_level


# ── Swivel Bearing Connection Builder ──────────────────────
def wire_swivel_bearings(sub_levels, joints):
    """
    For each bar's swivel bearings, set SubLevelID and SwivelPlate
    to reference the connected bar's sub_level and the bearing's
    local position on that connected bar.
    
    The key insight: a swivel bearing on bar X at joint J references
    the connected bar Y's UUID and the local position of bar Y's
    bearing at the same joint J.
    """
    # Build a map: joint_index -> list of (bar_name, bearing_local_pos, bearing_index_in_blocks)
    joint_to_bearings = {}
    for sl in sub_levels:
        meta = sl['_meta']
        bar = meta['bar_name']
        ja = meta['joint_a_idx']
        jb = meta['joint_b_idx']
        
        # Find which bearing is at joint_a and which at joint_b
        for blk in sl['blocks']:
            if blk['state'] in (1, 2) and 'nbt' in blk:
                local_pos = blk['pos']
                if local_pos == meta['bearing_a_local']:
                    joint_to_bearings.setdefault(ja, []).append({
                        'bar': bar, 'local': local_pos, 'block': blk,
                        'sub_level': sl
                    })
                elif local_pos == meta['bearing_b_local']:
                    joint_to_bearings.setdefault(jb, []).append({
                        'bar': bar, 'local': local_pos, 'block': blk,
                        'sub_level': sl
                    })
    
    # Wire each bearing: for bearings at the same joint, they reference each other
    for joint_idx, bearings in joint_to_bearings.items():
        if len(bearings) < 2:
            continue  # endpoint bearing (only one bar connected)
        
        for i, bear in enumerate(bearings):
            # Find the OTHER bar's bearing at this joint
            for j, other in enumerate(bearings):
                if i != j:
                    bear['block']['nbt']['SubLevelID'] = other['sub_level']['uuid']
                    bear['block']['nbt']['SwivelPlate'] = other['local']
                    break


# ── Compute schematic bounding box ─────────────────────────
def compute_size(sub_levels):
    """Compute the overall size of the schematic."""
    mins = [float('inf'), float('inf'), float('inf')]
    maxs = [float('-inf'), float('-inf'), float('-inf')]
    
    for sl in sub_levels:
        pos = [sl['position']['x'], sl['position']['y'], sl['position']['z']]
        half = [s / 2 for s in sl['size']]
        for axis in range(3):
            mins[axis] = min(mins[axis], pos[axis] - half[axis])
            maxs[axis] = max(maxs[axis], pos[axis] + half[axis])
    
    return [max(1, round(maxs[i] - mins[i]) + 1) for i in range(3)]


# ── Main Generator ─────────────────────────────────────────
def generate_schematic(lengths, crank_angle=0.0, output_path=None):
    """
    Generate a complete Minecraft schematic JSON for the Jansen linkage.
    
    Args:
        lengths: dict of bar lengths (a through m)
        crank_angle: crank angle in degrees (0-360)
        output_path: file path to write JSON (optional)
    
    Returns:
        dict: the complete schematic data
    """
    print(f"Solving linkage at crank angle = {crank_angle}° ...")
    joints = solve_linkage(crank_angle, lengths)
    
    # Print joint positions for debugging
    joint_names = ['J0(origin)', 'J1(crank)', 'J2(tip)', 'J3', 'J4', 'J5', 'J6', 'J7(foot)']
    for idx, (name, pos) in enumerate(zip(joint_names, joints)):
        print(f"  {name}: ({pos[0]:7.2f}, {pos[1]:7.2f})")
    
    # Build sub_levels for each bar
    sub_levels = []
    for bar_name in sorted(BAR_CONNECTIONS.keys()):
        ja, jb = BAR_CONNECTIONS[bar_name]
        sl = build_bar_sublevel(bar_name, joints[ja], joints[jb], lengths)
        sub_levels.append(sl)
        print(f"  Bar {bar_name}: {BAR_CONNECTIONS[bar_name][0]}->{BAR_CONNECTIONS[bar_name][1]}, "
              f"blocks={max(1, round(lengths[bar_name]*SCALE))}")
    
    # Wire swivel bearing connections
    wire_swivel_bearings(sub_levels, joints)
    
    # Compute bounding box
    size = compute_size(sub_levels)
    
    # Clean up internal metadata
    for sl in sub_levels:
        sl.pop('_meta', None)
    
    # Build root blocks (fixed pivot markers)
    root_blocks = []
    root_palette = [
        {'Name': 'create:brass_casing'},
        {'Name': 'simulated:swivel_bearing_link_block',
         'Properties': {'facing': 'north'}}
    ]
    
    # Add pivot markers at J0 and J1
    for joint_idx, pivot_name in [(0, 'J0'), (1, 'J1')]:
        px = round(joints[joint_idx][0] * SCALE + ORIGIN['x'], 8)
        py = round(joints[joint_idx][1] * SCALE + ORIGIN['y'], 8)
        root_blocks.append({
            'pos': [round(px), round(py), round(ORIGIN['z'])],
            'state': 1,
            'nbt': {
                'Speed': 0.0, 'NeedsSpeedUpdate': 1,
                'id': 'simulated:swivel_bearing_link_block'
            }
        })
    
    # Assemble the schematic
    schematic = {
        'size': size,
        'entities': [],
        'blocks': root_blocks,
        'sub_levels': sub_levels,
        'palette': root_palette,
        'DataVersion': 3955,
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schematic, f, indent=2, ensure_ascii=False)
        print(f"\nSchematic written to {output_path}")
    
    return schematic


# ── CLI Entry Point ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate Jansen linkage Minecraft schematic')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON file with custom bar lengths')
    parser.add_argument('--angle', type=float, default=0.0,
                        help='Crank angle in degrees (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: scripts/jansen_schematic_<angle>.json)')
    args = parser.parse_args()
    
    # Load bar lengths
    lengths = dict(DEFAULT_LENGTHS)
    if args.config:
        print(f"Loading bar lengths from {args.config} ...")
        with open(args.config, 'r') as f:
            custom = json.load(f)
        lengths.update(custom)
    
    # Print lengths
    print("Bar lengths:")
    for name in sorted(lengths.keys()):
        print(f"  {name} = {lengths[name]:.1f}")
    
    # Output path
    if not args.output:
        safe_angle = str(args.angle).replace('.', '_')
        args.output = f"scripts/jansen_schematic_{safe_angle}.json"
    
    # Generate
    schematic = generate_schematic(lengths, args.angle, args.output)
    
    # Summary
    print(f"\nSchematic summary:")
    print(f"  Size: {schematic['size']}")
    print(f"  Sub-levels: {len(schematic['sub_levels'])}")
    print(f"  Root blocks: {len(schematic['blocks'])}")


if __name__ == '__main__':
    main()
