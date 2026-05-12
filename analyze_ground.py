import numpy as np
from optimize_integer import compute_foot_path_jit, ORIGINAL

L = np.array([ORIGINAL[k] for k in ['a','b','c','d','e','f','g','h','i','j','k','l','m']])
path, _ = compute_foot_path_jit(L, 180)
y = path[:, 1]
angles_deg = np.arange(180) * 2.0  # 0..358 deg

y_min = y.min()
y_range = y.max() - y.min()
ground_mask = y <= y_min + 0.2 * y_range
ground_angles = angles_deg[ground_mask]

print(f"Y range: [{y.min():.2f}, {y.max():.2f}], span={y_range:.2f}")
print(f"Ground contact: {ground_mask.sum()}/180 points")
print(f"  Angle range: {ground_angles[0]:.0f}–{ground_angles[-1]:.0f} deg")
print(f"  First 5: {ground_angles[:5]}")
print(f"  Last 5:  {ground_angles[-5:]}")

# Flatness = std(Y) during ground contact
ground_y = y[ground_mask]
print(f"\nGround Y: mean={ground_y.mean():.3f}, std={ground_y.std():.3f}, range={ground_y.max()-ground_y.min():.3f}")

# dY/d(theta) during ground contact
dy = np.gradient(y, angles_deg)
ground_dy = dy[ground_mask]
print(f"dY/d(theta): mean={ground_dy.mean():.5f}, std={ground_dy.std():.5f}, max_abs={np.abs(ground_dy).max():.5f}")

# What about the top of the cycle?
top_mask = y >= y_min + 0.8 * y_range
top_angles = angles_deg[top_mask]
print(f"\nTop of cycle: {top_mask.sum()}/180 points, angles {top_angles[0]:.0f}–{top_angles[-1]:.0f} deg")
