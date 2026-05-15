"""
Jansen Linkage Integer Optimization
=====================================
Given that Jansen linkages are very sensitive to bar length changes, this script
exhaustively searches all integer perturbations of the original bar lengths (±1 unit)
to find the best matches to the ideal foot path shape.
This is useful for building Jansen linkages in Minecraft, Lego, or other discrete media.

Reports two rankings for every run:
  • Best Shape Match  — closest to the original foot path shape
  • Flattest Ground   — flattest ground-contact phase of the cycle

Both shape and flatness are always tracked in Pass 1 and reported in the final output.

Key performance strategies:
  1. Numba JIT compilation of the solver (10-50x speedup)
  2. Aggressive geometric pruning (reject impossible combos before solving) Most common with very small ratios like 0.1x where some bars become = 0
  3. Continuation solving (neighbouring angles share initial guess)
  4. Two-pass evaluation (cheap pass filters 90%+ of combinations)
  5. Dual min-heap top-N tracking (shape + flatness, never materialize full list)

Usage Examples:
    # Default run (both metrics tracked and reported)
    python optimize_integer.py --scale 0.2

    # Show more results
    python optimize_integer.py --scale 0.2 --top 10

    # Quick test run (fewer angles, faster but coarser)
    python optimize_integer.py --scale 0.2 --quick

    # Full resolution at larger scale
    python optimize_integer.py --scale 1.0 --angles 360

    # Show ASCII art of the Jansen linkage alongside progress output (colored teal)
    python optimize_integer.py --scale 0.2 --asciiart

Arguments:
    --scale    Scale factor (default: 0.2, use 1.0 for 1:1, 10 for 10x)
    --angles   Full evaluation angles (default: 360)
    --top      Number of best combos to report (default: 3)
    --sample   Cheap-pass sample angles (default: 12)
    --quick    Quick mode: 6 sample angles, 180 full angles
    --export   Directory to export JSON config (default: .)
    --asciiart Show ASCII art of the Jansen linkage alongside progress output (teal).
               Art lines are vertically centered across progress updates. Disabled by default.
"""

import argparse
import sys
import time
import json
import os
import heapq
import numpy as np

# ── Try Numba for JIT acceleration ───────────────────────────
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators (no-op)
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else args[0]
    prange = range

# ── Original bar lengths ──────────────────────────────────────
ORIGINAL = {
    'a': 38.0, 'b': 41.5, 'c': 39.3, 'd': 40.1,
    'e': 55.8, 'f': 39.4, 'g': 36.7, 'h': 65.7,
    'i': 49.0, 'j': 50.0, 'k': 61.9, 'l': 7.8, 'm': 15.0,
}
BAR_KEYS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
NUM_BARS = 13

def _ptp(arr):
    """Peak-to-peak (NumPy 2.x compatibility — .ptp() was removed)"""
    return arr.max() - arr.min()

# ── Numba-accelerated solver ─────────────────────────────────
# All solver internals use plain arrays for Numba compatibility.
# Bar lengths are passed as a 13-element array indexed by position
# in BAR_KEYS: a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7,
#              i=8, j=9, k=10, l=11, m=12

@njit(cache=True)
def compute_residuals(x, L, J2):
    """
    Compute 10 constraint residuals.
    x: 10-element array [J3x,J3y,J4x,J4y,J5x,J5y,J6x,J6y,J7x,J7y]
    L: 13-element array of bar lengths (indexed by BAR_KEYS position)
    J2: 2-element array [x,y] (crank tip, known from angle)
    Returns: (residuals_10, cost)
    """
    # J0 = [0,0], J1 = [L[a], L[l]]
    J0x, J0y = 0.0, 0.0
    J1x, J1y = L[0], L[11]  # a, l

    J3x, J3y = x[0], x[1]
    J4x, J4y = x[2], x[3]
    J5x, J5y = x[4], x[5]
    J6x, J6y = x[6], x[7]
    J7x, J7y = x[8], x[9]

    r = np.empty(10)
    # b(1): J0-J3
    r[0] = np.sqrt((J3x-J0x)**2 + (J3y-J0y)**2) - L[1]
    # j(9): J2-J3
    r[1] = np.sqrt((J3x-J2[0])**2 + (J3y-J2[1])**2) - L[9]
    # e(4): J3-J4
    r[2] = np.sqrt((J4x-J3x)**2 + (J4y-J3y)**2) - L[4]
    # d(3): J0-J4
    r[3] = np.sqrt((J4x-J0x)**2 + (J4y-J0y)**2) - L[3]
    # f(5): J4-J5
    r[4] = np.sqrt((J5x-J4x)**2 + (J5y-J4y)**2) - L[5]
    # c(2): J0-J6
    r[5] = np.sqrt((J6x-J0x)**2 + (J6y-J0y)**2) - L[2]
    # k(10): J2-J6
    r[6] = np.sqrt((J6x-J2[0])**2 + (J6y-J2[1])**2) - L[10]
    # g(6): J5-J6
    r[7] = np.sqrt((J6x-J5x)**2 + (J6y-J5y)**2) - L[6]
    # h(7): J5-J7
    r[8] = np.sqrt((J7x-J5x)**2 + (J7y-J5y)**2) - L[7]
    # i(8): J6-J7
    r[9] = np.sqrt((J7x-J6x)**2 + (J7y-J6y)**2) - L[8]

    cost = 0.0
    for ii in range(10):
        cost += r[ii] * r[ii]
    return r, cost


@njit(cache=True)
def solve_single_angle(theta, L, guess):
    """
    Solve Jansen linkage at a single crank angle using
    damped Gauss-Newton with numerical Jacobian.

    Parameters:
        theta: crank angle (radians)
        L: 13-element array of bar lengths
        guess: 10-element initial guess (J3..J7 positions)
    Returns:
        (x_final, converged, cost, J7x, J7y)
    """
    J0x, J0y = 0.0, 0.0
    J1x, J1y = L[0], L[11]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    J2x = J1x + L[12] * cos_t
    J2y = J1y + L[12] * sin_t
    J2 = np.array([J2x, J2y])

    x = guess.copy()
    lam = 1e-4
    tol = 1e-8
    max_iter = 40

    for it in range(max_iter):
        r, cost = compute_residuals(x, L, J2)

        if cost < tol:
            return x, True, cost, x[8], x[9]

        # Central-difference Jacobian (10×10)
        J = np.empty((10, 10))
        for k in range(10):
            h = 1e-6 * (abs(x[k]) + 1.0)
            x[k] += h
            fp = np.empty(10)
            fp, _ = compute_residuals(x, L, J2)
            x[k] -= 2 * h
            fm = np.empty(10)
            fm, _ = compute_residuals(x, L, J2)
            x[k] += h
            for ii in range(10):
                J[ii, k] = (fp[ii] - fm[ii]) / (2 * h)

        # Jt * r
        Jtr = np.empty(10)
        for k in range(10):
            s = 0.0
            for ii in range(10):
                s += J[ii, k] * r[ii]
            Jtr[k] = s

        # Gradient norm
        gn = 0.0
        for k in range(10):
            gn += Jtr[k] * Jtr[k]
        if gn < 1e-16:
            return x, cost < 1e-4, cost, x[8], x[9]

        # JtJ (10×10)
        JtJ = np.empty((10, 10))
        for i in range(10):
            for jj in range(10):
                s = 0.0
                for kk in range(10):
                    s += J[kk, i] * J[kk, jj]
                JtJ[i, jj] = s

        # Damped: (JtJ + lam*diag(JtJ)) * dx = -Jtr
        A = np.empty((10, 10))
        for i in range(10):
            for jj in range(10):
                if i == jj:
                    A[i, jj] = JtJ[i, jj] + lam * (JtJ[i, jj] + 1e-12)
                else:
                    A[i, jj] = JtJ[i, jj]

        # Solve via LU (numpy linalg)
        try:
            dx = np.linalg.solve(A, -Jtr)
        except:
            lam *= 10.0
            continue

        # Trial step
        xp = x + dx
        rp, costp = compute_residuals(xp, L, J2)

        if costp < cost:
            x = xp
            r = rp
            cost = costp
            lam = max(lam * 0.3, 1e-12)
            # Convergence check
            max_rel = 0.0
            for k in range(10):
                rel = abs(dx[k]) / (abs(x[k]) + 1.0)
                if rel > max_rel:
                    max_rel = rel
            if max_rel < 1e-10:
                return x, True, cost, x[8], x[9]
        else:
            lam *= 10.0
            if lam > 1e12:
                break

    return x, cost < 1e-4, cost, x[8], x[9]


@njit(cache=True)
def make_default_guess(L):
    """Generate default initial guess for solver."""
    x = np.empty(10)
    x[0] = L[1] * 0.3   # J3x
    x[1] = L[1] * 0.9   # J3y
    x[2] = -L[3] * 0.7  # J4x
    x[3] = L[3] * 0.7   # J4y
    x[4] = -L[5] * 0.8  # J5x
    x[5] = 0.0          # J5y
    x[6] = L[2] * 0.5   # J6x
    x[7] = -L[2] * 0.8  # J6y
    x[8] = -L[7] * 0.3  # J7x
    x[9] = -L[7] * 0.9  # J7y
    return x


@njit(cache=True)
def compute_foot_path_jit(L, num_angles):
    """
    Compute foot path (J7 trajectory) at num_angles evenly-spaced
    crank angles using continuation solving.

    Returns: (foot_path Nx2, all_converged)
    """
    foot_path = np.empty((num_angles, 2))
    guess = make_default_guess(L)
    all_converged = True

    for idx in range(num_angles):
        theta = 2.0 * np.pi * idx / num_angles
        x, converged, cost, J7x, J7y = solve_single_angle(theta, L, guess)
        foot_path[idx, 0] = J7x
        foot_path[idx, 1] = J7y
        if converged:
            guess = x  # carry solution forward as initial guess for next angle
        if not converged:
            all_converged = False

    return foot_path, all_converged


@njit(cache=True)
def check_geometry_fast(L):
    """
    Fast geometric feasibility check using static constraints.

    L: 13-element array of bar lengths
    Returns: True if configuration passes all checks
    """
    a, b, c, d, e, f, g, h, i, j, k, ll, m = L

    # All bars must be positive
    if a <= 0 or b <= 0 or c <= 0 or d <= 0 or e <= 0 or f <= 0:
        return False
    if g <= 0 or h <= 0 or i <= 0 or j <= 0 or k <= 0 or ll < 0 or m <= 0:
        return False

    # Triangle J5-J6-J7: g + h > i, g + i > h, h + i > g
    if g + h <= i or g + i <= h or h + i <= g:
        return False

    # Loop J0-J3-J4: triangle inequalities
    if e >= b + d:
        return False
    if b >= e + d:
        return False
    if d >= b + e:
        return False

    return True


@njit(cache=True)
def compute_flatness(foot_path):
    """
    Compute flatness of ground contact phase.

    Ground contact = angles where Y <= min_Y + 0.2 * Y_range
    Flatness        = std(Y) during ground contact (lower = flatter)

    Returns flatness value, or 1e10 if no ground points found.
    """
    n = foot_path.shape[0]
    if n == 0:
        return 1e10

    # Find Y min and range
    y_min = foot_path[0, 1]
    y_max = foot_path[0, 1]
    for i in range(1, n):
        if foot_path[i, 1] < y_min:
            y_min = foot_path[i, 1]
        if foot_path[i, 1] > y_max:
            y_max = foot_path[i, 1]
    y_range = y_max - y_min
    if y_range < 1e-10:
        return 0.0  # perfectly flat (degenerate)

    # Ground threshold
    threshold = y_min + 0.2 * y_range

    # Collect ground Y values
    ground_sum = 0.0
    ground_count = 0
    for i in range(n):
        if foot_path[i, 1] <= threshold:
            ground_sum += foot_path[i, 1]
            ground_count += 1

    if ground_count < 2:
        return 1e10

    ground_mean = ground_sum / ground_count

    # Compute std
    ss = 0.0
    for i in range(n):
        if foot_path[i, 1] <= threshold:
            diff = foot_path[i, 1] - ground_mean
            ss += diff * diff
    flatness = np.sqrt(ss / ground_count)

    # Normalize by Y range so it's scale-independent (percentage)
    return flatness / y_range * 100.0


# ── Path comparison metric ───────────────────────────────────

def compare_paths(ref_path, test_path):
    """
    Compare two foot paths with scale & translation normalization.
    Returns composite error score (lower = better match).
    """
    ref = ref_path.astype(np.float64).copy()
    tst = test_path.astype(np.float64).copy()

    # Center both paths
    ref = ref - ref.mean(axis=0)
    tst = tst - tst.mean(axis=0)

    # Scale test to match reference size (bounding box diagonal)
    ref_size = np.sqrt(_ptp(ref[:, 0])**2 + _ptp(ref[:, 1])**2)
    tst_size = np.sqrt(_ptp(tst[:, 0])**2 + _ptp(tst[:, 1])**2)
    if tst_size < 1e-10:
        return 1e10
    tst = tst * (ref_size / tst_size)

    # Angle-by-angle RMS error (normalized by reference size → percentage)
    rms = np.sqrt(np.mean((ref - tst)**2))
    if ref_size < 1e-10:
        return 1e10
    rms_pct = rms / ref_size  # percentage of reference path size

    if rms_pct > 0.05:  # >5% deviation
        return rms_pct * 100 + 100  # early reject for very bad paths

    # ── Gait quality metrics ──
    # Height range
    ref_y_range = _ptp(ref[:, 1])
    tst_y_range = _ptp(tst[:, 1])
    if ref_y_range > 1e-10:
        height_err = abs(ref_y_range - tst_y_range) / ref_y_range
    else:
        height_err = 1.0

    # Horizontal stride (X-range)
    ref_x_range = _ptp(ref[:, 0])
    tst_x_range = _ptp(tst[:, 0])
    if ref_x_range > 1e-10:
        stride_err = abs(ref_x_range - tst_x_range) / ref_x_range
    else:
        stride_err = 1.0

    # Flat-top fraction: % of cycle where foot Y is near minimum
    ground_frac = 0.2
    ref_ground = np.sum(ref[:, 1] <= ref[:, 1].min() + ground_frac * ref_y_range) / len(ref)
    tst_ground = np.sum(tst[:, 1] <= tst[:, 1].min() + ground_frac * tst_y_range) / len(tst)
    ground_err = abs(ref_ground - tst_ground)

    # Composite score
    return rms_pct * 100 + height_err * 30 + stride_err * 30 + ground_err * 20


# ── Numba-compatible path comparison (for JIT path) ──────────

@njit(cache=True)
def compare_paths_simple(ref_path, test_path):
    """Simplified path comparison for Numba compilation."""
    n = ref_path.shape[0]

    # Compute centroids
    ref_cx, ref_cy = 0.0, 0.0
    tst_cx, tst_cy = 0.0, 0.0
    for i in range(n):
        ref_cx += ref_path[i, 0]
        ref_cy += ref_path[i, 1]
        tst_cx += test_path[i, 0]
        tst_cy += test_path[i, 1]
    ref_cx /= n; ref_cy /= n
    tst_cx /= n; tst_cy /= n

    # Center and compute bounding box sizes
    ref_xmin, ref_xmax = ref_path[0,0] - ref_cx, ref_path[0,0] - ref_cx
    ref_ymin, ref_ymax = ref_path[0,1] - ref_cy, ref_path[0,1] - ref_cy
    tst_xmin, tst_xmax = test_path[0,0] - tst_cx, test_path[0,0] - tst_cx
    tst_ymin, tst_ymax = test_path[0,1] - tst_cy, test_path[0,1] - tst_cy

    for i in range(1, n):
        rx = ref_path[i, 0] - ref_cx
        ry = ref_path[i, 1] - ref_cy
        tx = test_path[i, 0] - tst_cx
        ty = test_path[i, 1] - tst_cy
        if rx < ref_xmin: ref_xmin = rx
        if rx > ref_xmax: ref_xmax = rx
        if ry < ref_ymin: ref_ymin = ry
        if ry > ref_ymax: ref_ymax = ry
        if tx < tst_xmin: tst_xmin = tx
        if tx > tst_xmax: tst_xmax = tx
        if ty < tst_ymin: tst_ymin = ty
        if ty > tst_ymax: tst_ymax = ty

    ref_size = np.sqrt((ref_xmax - ref_xmin)**2 + (ref_ymax - ref_ymin)**2)
    tst_size = np.sqrt((tst_xmax - tst_xmin)**2 + (tst_ymax - tst_ymin)**2)
    if tst_size < 1e-10:
        return 1e10

    scale = ref_size / tst_size

    # Scaled RMS
    ss = 0.0
    for i in range(n):
        dx = (ref_path[i, 0] - ref_cx) - (test_path[i, 0] - tst_cx) * scale
        dy = (ref_path[i, 1] - ref_cy) - (test_path[i, 1] - tst_cy) * scale
        ss += dx*dx + dy*dy
    rms = np.sqrt(ss / n)

    # Normalize by reference path size so scores are scale-independent (percentage)
    # ref_size = diagonal of the bounding box of the centered reference path
    if ref_size < 1e-10:
        return 1e10
    return (rms / ref_size) * 100.0


# ── Combination enumeration ──────────────────────────────────

def dict_to_array(lengths_dict):
    """Convert {key: length} dict to 13-element array."""
    return np.array([lengths_dict[k] for k in BAR_KEYS])


def array_to_dict(L):
    """Convert 13-element array to {key: length} dict."""
    return {k: L[i] for i, k in enumerate(BAR_KEYS)}


def enumerate_combinations(base_lengths_dict, scale, top_n=3,
                           full_angles=360, sample_angles=12, show_asciiart=False):
    """
    Main enumeration: search all 3^13 perturbations.
    """
    base_L = dict_to_array(base_lengths_dict)
    base_int_L = np.round(base_L * scale).astype(int)

    print(f"Integer baseline: {array_to_dict(base_int_L)}")
    print()

    # ── Compute reference path from scaled (float) lengths ──
    # This is the ideal shape at the target scale; integer perturbations are compared against it
    scaled_L = base_L * scale
    print("Computing reference foot path (scaled float lengths)...")
    print(f"  Scaled lengths: {array_to_dict(np.round(scaled_L, 1))}")
    t0 = time.time()
    ref_path_full = compute_foot_path_jit(scaled_L.astype(np.float64), full_angles)
    print(f"  Done in {time.time()-t0:.1f}s, "
          f"converged={ref_path_full[1]}, shape={ref_path_full[0].shape}")

    # ── Compute baseline integer path ──
    print("Computing integer baseline foot path...")
    t0 = time.time()
    baseline_result = compute_foot_path_jit(base_int_L.astype(np.float64), full_angles)
    baseline_path = baseline_result[0]
    baseline_score = compare_paths_simple(ref_path_full[0], baseline_path)
    baseline_flat = compute_flatness(baseline_path)
    print(f"  Done in {time.time()-t0:.1f}s, score={baseline_score:.4f}, "
          f"flatness={baseline_flat:.4f}, converged={baseline_result[1]}")

    # Downsample reference for quick pass
    ref_path_array = ref_path_full[0]  # extract path from (path, converged) tuple
    ref_sample = np.empty((sample_angles, 2))
    for i in range(sample_angles):
        idx = int(i * full_angles / sample_angles) % full_angles
        ref_sample[i] = ref_path_array[idx]

    # ── Enumerate all 3^13 combinations ──
    total = 3 ** NUM_BARS
    print(f"\nEnumerating {total:,} combinations (3^{NUM_BARS})...")
    print(f"  Sample pass: {sample_angles} angles | Full pass: {full_angles} angles")
    print(f"  Numba: {'enabled' if HAS_NUMBA else 'disabled (install numba for ~50x speedup)'}")
    print()

    heap_size = max(top_n * 20, 100)

    # Min-heap stores (neg_score, idx, perturbation_tuple) — L is reconstructed from perturb in Pass 2
    heap = []  # shape-match heap

    # Second heap for flatness tracking
    flat_heap = []  # (-flatness, idx, perturbation_tuple)

    eval_count = 0
    prune_count = 0
    nan_count = 0
    converge_count = 0
    best_shape = float('inf')
    best_flat = float('inf')
    within_5pct_shape = 0
    within_5pct_flat = 0
    t_start = time.time()
    last_report = 0

    # ASCII art progress: center art lines vertically across progress outputs
    art_len = len(LINKAGE_ART)
    estimated_progress_lines = (total // 25_000) + 1
    art_offset = max(0, (estimated_progress_lines - art_len) // 2)
    progress_line = 0  # increments each time we print progress
     # show_asciiart is passed as a parameter from main()

    # Precompute base_int_L as float for solver
    base_int_f = base_int_L.astype(np.float64)

    # Iterative odometer: perturbions cycle through [-1, 0, 1] per bar
    # Start at [-1, -1, ..., -1] (index 1, skipping all-zero at index 0)
    perturb = np.full(NUM_BARS, -1, dtype=np.int32)

    # Account for the skipped all-zero combination
    eval_count += 1

    for idx in range(1, total):
        # Build perturbed lengths
        L = base_int_f + perturb.astype(np.float64)

        # Fast geometric pruning
        if not check_geometry_fast(L):
            prune_count += 1

            # Advance odometer
            for b in range(NUM_BARS - 1, -1, -1):
                perturb[b] += 1
                if perturb[b] <= 1:
                    break
                perturb[b] = -1
            continue

        # ── Pass 1: cheap sample evaluation ──
        result = compute_foot_path_jit(L, sample_angles)
        foot = result[0]

        # Check for NaN (solver failure)
        if np.any(np.isnan(foot)):
            nan_count += 1

            # Advance odometer
            for b in range(NUM_BARS - 1, -1, -1):
                perturb[b] += 1
                if perturb[b] <= 1:
                    break
                perturb[b] = -1
            continue

        # Track convergence (all frames converged)
        if result[1]:
            converge_count += 1

        # Quick score
        score = compare_paths_simple(ref_sample, foot)
        perturb_tuple = tuple(perturb)

        # Maintain shape-match heap (store perturb only, reconstruct L in Pass 2)
        neg_score = -score
        if len(heap) < heap_size:
            heapq.heappush(heap, (neg_score, idx, perturb_tuple))
        elif neg_score > heap[0][0]:
            heapq.heapreplace(heap, (neg_score, idx, perturb_tuple))

        # Track within 5% of best shape
        if score < best_shape:
            best_shape = score
        if score <= best_shape * 1.05:
            within_5pct_shape += 1

        # Maintain flatness heap (always on)
        flatness = compute_flatness(foot)

        # Track within 5% of best flatness
        if flatness < best_flat:
            best_flat = flatness
        if flatness <= best_flat * 1.05:
            within_5pct_flat += 1

        neg_flatness = -flatness
        if len(flat_heap) < heap_size:
            heapq.heappush(flat_heap, (neg_flatness, idx, perturb_tuple))
        elif neg_flatness > flat_heap[0][0]:
            heapq.heapreplace(flat_heap, (neg_flatness, idx, perturb_tuple))

        eval_count += 1

        # Progress reporting (every 25,000 combos)
        if eval_count - last_report >= 25_000:
            elapsed = time.time() - t_start
            rate = eval_count / elapsed if elapsed > 0 else 0
            eta = (total - eval_count) / rate if rate > 0 else 0
            best_score = -heap[0][0] if heap else float('inf')
            best_flat = -flat_heap[0][0] if flat_heap else float('inf')
            line = (f"  [{eval_count:,}/{total:,}] "
                  f"{eval_count/total*100:.1f}% | "
                  f"{rate:,.0f} combos/sec | "
                  f"ETA {eta/60:.1f}m | "
                  f"Best Shape: {best_score:.4f} | Best Flatness: {best_flat:.4f}")
            line += f" | Pruned: {prune_count:,} | NaN: {nan_count:,}"

            # Combine with ASCII art line on the same line (art on the right, fixed column, colored teal)
            progress_width = 108  # width of the progress text before the art starts
            art_idx = progress_line - art_offset
            if show_asciiart and 0 <= art_idx < art_len:
                art_line = _CYAN + LINKAGE_ART[art_idx] + _RESET
                padded = line.ljust(progress_width)
                print(f"  {padded}  {art_line}")
            else:
                print(f"  {line}")
            last_report = eval_count
            progress_line += 1

        # Advance odometer for next iteration
        for b in range(NUM_BARS - 1, -1, -1):
            perturb[b] += 1
            if perturb[b] <= 1:
                break
            perturb[b] = -1

    elapsed_pass1 = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pass 1 complete in {elapsed_pass1:.1f}s")
    print(f"  Evaluated: {eval_count:,} | Pruned: {prune_count:,} | NaN: {nan_count:,}")
    print(f"  Top {len(heap)} candidates tracked")
    print(f"{'='*60}")

    # ── Merge shape + flatness heaps for Pass 2 ──
    if flat_heap:
        # Deduplicate by idx (perturbation index)
        seen = set()
        merged = []
        for item in heap + flat_heap:
            neg_s, item_idx, p = item  # L is reconstructed from p in Pass 2
            if item_idx not in seen:
                seen.add(item_idx)
                merged.append(item)
        heap = merged
        print(f"  Merged heaps: {len(merged)} unique candidates (shape: {len(heap)-len(flat_heap)}, flat: {len(flat_heap)})")

    print(f"\nPass 2: Full {full_angles}-angle evaluation of top {len(heap)} candidates...")
    t2 = time.time()

    # Sort heap by score (best first)
    heap.sort(key=lambda x: x[0])  # smallest -score first = best score first

    full_results = []
    for rank, (neg_score, idx, perturb) in enumerate(heap):
        # Reconstruct L from perturbation tuple
        L = base_int_f + np.array(perturb, dtype=np.float64)
        result = compute_foot_path_jit(L, full_angles)
        foot = result[0]
        if np.any(np.isnan(foot)):
            continue
        full_score = compare_paths(ref_path_full[0], foot)
        flatness = compute_flatness(foot)
        full_results.append((full_score, perturb, L, foot, result[1], flatness))

    # Sort by full score
    full_results.sort(key=lambda x: x[0])
    elapsed_pass2 = time.time() - t2

    # Add baseline
    baseline_tuple = (baseline_score, tuple(np.zeros(NUM_BARS, dtype=np.int32)),
                     base_int_f.copy(), baseline_path, baseline_result[1], baseline_flat)
    full_results.insert(0, baseline_tuple)
    full_results.sort(key=lambda x: x[0])

    print(f"Pass 2 complete in {elapsed_pass2:.1f}s")

    # Filter to only converged combinations (both shape and flatness must converge)
    full_results_converged = [r for r in full_results if r[4]]

    top_n_results = full_results_converged[:top_n]
    # Find top-N by flatness (always computed, converged only)
    full_results_by_flat = sorted(full_results_converged, key=lambda x: x[5])  # sort by flatness
    top_flat_results = full_results_by_flat[:top_n]

    return top_n_results, full_results, baseline_score, ref_path_full[0], base_int_f, top_flat_results, converge_count, eval_count, within_5pct_shape, within_5pct_flat, estimated_progress_lines


# ── Reporting ────────────────────────────────────────────────

# ANSI color helpers
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

# ASCII art of the Jansen linkage (51 lines, printed alongside progress output)
LINKAGE_ART = [
    "                                     .",
    "                               .-+++  -..",
    "                           .-+++  .   ..-..",
    "                       .-+++     ##  +-...--.",
    "                   .-+++     -++--  -...-...--..",
    "               .-++- .   +++-.. .. -.   ..-... -..",
    "           .-++- .   ++-..     .-  -.     ..-....--.....",
    "       .-+++ ..  ++-..         -. -.        ..--.. .    ....",
    "  ..-##- ..  ++-...           -+ +.        ..   .-...--.   ..",
    "  .  ..  ++-..               .- .-        ..  .....--.. --.  ..",
    "  .   . -++-..              .-  -.        .  .     ..--...--. .",
    "  .-. #-  . .+++..          --  ..       .. .     .---.--...-.-.",
    "   .-  --++-- .  +++-.     .-  -.        . ..     .   .   .-.  .",
    "    ..  ..  .+++. .  +++-..-  -.         .  .     .- -.....-  -.",
    "    ..   ..     .-++- .. .++  ...................... . ...  ...",
    "     ..  ...        .-++-    ..   .                  ...   -...",
    "      .-  -.            .-+  +.........................  -. ..",
    "       .-  -.             .-  -.            ..     ..  ... ..",
    "        ..  -.             -  ..             ......   .....",
    "        ...  ..            .-  -.              ...  -..",
    "         ..  ..             -- ..            ...  ...",
    "          .-  +-.           .-  -.         ...   ...",
    "           .- .---..         .+ ..        ..   -..",
    "            .+ .   +--.       -  -      ..-  ...",
    "             -  ++    -+-..   .+ +.   ...  ...",
    "             .. -..--.   .---. -. +....   ...",
    "             .-  .  ..-+-    ---+ --..  ...",
    "              .  -.    ..--+    ++ #.  ...",
    "              ..  .        ..--- .  ....",
    "              ..  -.          ..-# --.",
    "               ..  .            -  -.",
    "               ..  -.          .- -.",
    "                ..  .         .- -.",
    "                ..  -.       .-  -",
    "                 ..  .       .. ..",
    "                 .-  -.     .-  .",
    "                  ..  .    ..  -.",
    "                  .-  ..   .  ..",
    "                   .. ..  .-  ..",
    "                   .- .. .-  -.",
    "                    .. -..  ..",
    "                    .- --.  ..",
    "                     .. +  -.",
    "                     .- - -.",
    "                      .- ..",
    "                      .-.-.",
    "                       ....",
]

def format_perturbation(perturb, base_int):
    """Format perturbation tuple as readable colored string."""
    parts = []
    for b, key in enumerate(BAR_KEYS):
        orig = int(round(base_int[b]))
        delta = perturb[b]
        perturbed = orig + delta
        if delta > 0:
            parts.append(f"{_YELLOW}{key}={perturbed}{_GREEN}(+{delta}){_RESET}{_RESET}")
        elif delta < 0:
            parts.append(f"{_YELLOW}{key}={perturbed}{_CYAN}({delta}){_RESET}{_RESET}")
        else:
            parts.append(f"{_YELLOW}{key}={perturbed}{_RESET}")
    return ", ".join(parts)


def print_results(results, base_int, top_n=3, flat_results=None, converge_count=0, eval_count=0, within_5pct_shape=0, within_5pct_flat=0):
    """Print ranked results by shape-match and flatness."""
    print("\n" + "=" * 80)
    print(f"  {_GREEN}TOP RESULTS — Best Shape Match (Lower score is better){_RESET}")
    print("=" * 80)

    for rank, (score, perturb, L, foot, converged, flatness) in enumerate(results[:top_n]):
        is_baseline = all(p == 0 for p in perturb)
        label = " (BASELINE)" if is_baseline else ""
        conv_str = f"{_GREEN}✓ GOOD{_RESET}" if converged else f"{_RED}✗ DOES NOT CONVERGE{_RESET}"

        print(f"\n  #{rank + 1}{label}  —  {_GREEN}{_BOLD}Shape: {score:.4f}{_RESET}  Flatness: {flatness:.4f}  {conv_str}")
        print(f"  Bars: {format_perturbation(perturb, base_int)}")

        fp = foot
        x_range = _ptp(fp[:, 0])
        y_range = _ptp(fp[:, 1])
        ground_pct = 0.0
        if y_range > 1e-6:
            ground_pct = np.sum(fp[:, 1] <= fp[:, 1].min() + 0.2 * y_range) / len(fp) * 100
        print(f"  Foot path: "
              f"X: [{fp[:,0].min():.1f}, {fp[:,0].max():.1f}] stride ≈ {x_range:.1f} "
              f"- "
              f"Y: [{fp[:,1].min():.1f}, {fp[:,1].max():.1f}] lift ≈ {y_range:.1f} "
              f"- "
              f"Ground contact: ~{ground_pct:.0f}%")

    # Show top flatness results
    print("\n" + "=" * 80)
    print(f"  {_CYAN}TOP RESULTS — Flattest Ground Contact (Lower score is better){_RESET}")
    print("=" * 80)
    for rank, (score, perturb, L, foot, converged, flatness) in enumerate(flat_results[:top_n]):
        is_baseline = all(p == 0 for p in perturb)
        label = " (BASELINE)" if is_baseline else ""
        conv_str = f"{_GREEN}✓ GOOD{_RESET}" if converged else f"{_RED}✗ DOES NOT CONVERGE{_RESET}"

        print(f"\n  #{rank + 1}{label}  —  {_CYAN}{_BOLD}Flatness: {flatness:.4f}{_RESET}  Shape: {score:.4f}  {conv_str}")
        print(f"  Bars: {format_perturbation(perturb, base_int)}")

        fp = foot
        x_range = _ptp(fp[:, 0])
        y_range = _ptp(fp[:, 1])
        ground_pct = 0.0
        if y_range > 1e-6:
            ground_pct = np.sum(fp[:, 1] <= fp[:, 1].min() + 0.2 * y_range) / len(fp) * 100
        print(f"  Foot path: "
              f"X: [{fp[:,0].min():.1f}, {fp[:,0].max():.1f}] stride ≈ {x_range:.1f} "
              f"- "
              f"Y: [{fp[:,1].min():.1f}, {fp[:,1].max():.1f}] lift ≈ {y_range:.1f} "
              f"- "
              f"Ground contact: ~{ground_pct:.0f}%")

    # Convergence summary
    non_converging = eval_count - converge_count
    conv_pct = converge_count / eval_count * 100 if eval_count > 0 else 0
    non_conv_pct = non_converging / eval_count * 100 if eval_count > 0 else 0

    print("\n" + "=" * 80)
    print(f"  {_GREEN}CONVERGENCE SUMMARY{_RESET}")
    print("=" * 80)
    print(f"  {_GREEN}Converging:     {converge_count:>7,}  ({conv_pct:5.1f}%){_RESET}")
    shape_pct = within_5pct_shape / eval_count * 100 if eval_count > 0 else 0
    flat_pct = within_5pct_flat / eval_count * 100 if eval_count > 0 else 0
    print(f"\t  Within 5% of best shape:     {within_5pct_shape:>7,}  ({shape_pct:5.1f}%)")
    print(f"\t  Within 5% of best flatness:  {within_5pct_flat:>7,}  ({flat_pct:5.1f}%)")
    print(f"  {_RED}Non-converging: {non_converging:>7,}  ({non_conv_pct:5.1f}%){_RESET}")
    print(f"  Total evaluated:  {eval_count:>7,}{_RESET}")
    print("=" * 80)


# ── Export ───────────────────────────────────────────────────

def export_config(results, base_int, output_dir=".", flat_results=None):
    """Export top configs as JSON."""
    configs = []
    for rank, (score, perturb, L, foot, converged, flatness) in enumerate(results[:3]):
        config = {
            "rank": rank + 1,
            "score": round(float(score), 4),
            "flatness": round(float(flatness), 4),
            "converged": bool(converged),
            "lengths": {BAR_KEYS[i]: int(round(L[i])) for i in range(NUM_BARS)},
            "perturbations": {BAR_KEYS[i]: int(perturb[i]) for i in range(NUM_BARS)},
            "foot_path_stats": {
                "stride": round(float(_ptp(foot[:, 0])), 2),
                "lift": round(float(_ptp(foot[:, 1])), 2),
                "x_range": [round(float(foot[:, 0].min()), 2),
                           round(float(foot[:, 0].max()), 2)],
                "y_range": [round(float(foot[:, 1].min()), 2),
                           round(float(foot[:, 1].max()), 2)],
            }
        }
        configs.append(config)

    out_path = os.path.join(output_dir, "optimized_configs.json")
    export_data = {
        "baseline_integers": {BAR_KEYS[i]: int(round(base_int[i])) for i in range(NUM_BARS)},
        "top_configs": configs,
    }
    # Add top flatness configs
    flat_configs = []
    for rank, (score, perturb, L, foot, converged, flatness) in enumerate(flat_results[:3]):
        fc = {
            "rank": rank + 1,
            "score": round(float(score), 4),
            "flatness": round(float(flatness), 4),
            "converged": bool(converged),
            "lengths": {BAR_KEYS[i]: int(round(L[i])) for i in range(NUM_BARS)},
            "perturbations": {BAR_KEYS[i]: int(perturb[i]) for i in range(NUM_BARS)},
        }
        flat_configs.append(fc)
    export_data["top_flat_configs"] = flat_configs

    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nExported configs → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimize integer bar lengths for Jansen linkage")
    parser.add_argument("--scale", type=float, default=0.2,
                        help="Scale factor (default: 0.2 for 1/5th size, use 10 for 10×)")
    parser.add_argument("--angles", type=int, default=360,
                        help="Full evaluation angles (default: 360)")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of best combos to report (default: 3)")
    parser.add_argument("--sample", type=int, default=12,
                        help="Cheap-pass sample angles (default: 12)")
    parser.add_argument("--export", type=str, nargs="?", const=".", default=None,
                        help="Export JSON config (use with directory path, or no arg for current dir)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only 6 sample angles, smaller heap")
    parser.add_argument("--asciiart", action="store_true",
                        help="Show ASCII art of the Jansen linkage alongside progress output")
    args = parser.parse_args()

    if args.quick:
        args.sample = 6
        args.angles = 180

    print("=" * 50)
    print(f"  {_GREEN}{_BOLD}Jansen Linkage Integer Optimization{_RESET}")
    print("=" * 50)
    print(f"  {_YELLOW}Scale factor:{_RESET} {args.scale}")
    print(f"  {_YELLOW}Original:{_RESET} {ORIGINAL}")
    print(f"  {_YELLOW}Combinations:{_RESET} {3**NUM_BARS:,} (3^{NUM_BARS})")
    print(f"  {_YELLOW}Evaluation:{_RESET} {args.sample}-angle quick → {args.angles}-angle full")
    numba_status = f"{_GREEN}enabled{_RESET}" if HAS_NUMBA else f"{_RED}disabled — pip install numba{_RESET}"
    print(f"  {_YELLOW}Numba JIT:{_RESET} {numba_status}")
    print("=" * 50)
    print()

    top_results, all_results, baseline_score, ref_path, base_int, top_flat, converge_count, eval_count, within_5pct_shape, within_5pct_flat, est_lines = \
        enumerate_combinations(ORIGINAL, args.scale,
                               top_n=args.top,
                               full_angles=args.angles,
                               sample_angles=args.sample,
                               show_asciiart=args.asciiart)

    print_results(top_results, base_int, args.top, flat_results=top_flat,
                  converge_count=converge_count, eval_count=eval_count,
                  within_5pct_shape=within_5pct_shape, within_5pct_flat=within_5pct_flat)
    if args.export is not None:
        export_config(top_results, base_int, args.export, flat_results=top_flat)

    # Baseline ranking (among converged results only)
    baseline_rank = None
    all_converged = [r for r in all_results if r[4]]
    for i, (score, perturb, L, fp, conv, flt) in enumerate(all_converged):
        if all(p == 0 for p in perturb):
            baseline_rank = i + 1
            break
    if baseline_rank:
        print(f"\n  Baseline integer rounding ranked #{baseline_rank} "
              f"out of {len(all_converged)} converged combinations.")


if __name__ == "__main__":
    main()
