"""Smooth joint-space trajectories: a corner-free path through waypoints plus S-curve timing.

Pure numpy/scipy (no Isaac, no collision logic) so it can be unit-tested and reused
when the same joint path is later sent to a real controller.  Joint angles are
radians; ``dt`` is the sample period in seconds.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def arc_length(path):
    """Cumulative L2 joint-space distance along ``path`` (starts at 0)."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def polyline_path(waypoints, step_deg=0.25):
    """Dense straight-segment path through the waypoints (has corners)."""
    pts = _dedupe(waypoints)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.degrees(np.abs(b - a).max()) / step_deg)))
        dense += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.asarray(dense)


def smooth_path(waypoints, step_deg=0.25):
    """Dense, corner-free (C1) joint path through the waypoints.

    PCHIP is shape preserving: between two waypoints a joint never leaves the
    range spanned by them, so smoothing cannot push a joint past its limits.
    The first and last rows are exactly the first and last waypoints.
    """
    pts = _dedupe(waypoints)
    if len(pts) < 3:
        return polyline_path(pts, step_deg)
    u = arc_length(pts)
    travel = np.degrees(np.abs(np.diff(pts, axis=0)).max(axis=1)).sum()
    n = max(2, int(np.ceil(travel / step_deg)) + 1)
    return PchipInterpolator(u, pts, axis=0)(np.linspace(0.0, u[-1], n))


def time_parameterize(path, max_speed, max_accel, dt, smooth_time=0.4):
    """Joint positions every ``dt`` s along ``path`` within joint speed/accel limits.

    Time-optimal-style path parameterisation: the path speed is limited at every
    sample by (a) the fastest joint's speed limit and (b) the centripetal joint
    acceleration where the path bends, then a forward and a backward pass enforce
    the acceleration limit, so a gentle stretch runs at full speed while a sharp
    bend slows down only locally.  The positions are finally smoothed over
    ``smooth_time`` s (jerk limiting); the motion starts and ends at rest.
    ``max_speed`` is rad/s and ``max_accel`` rad/s^2 for every joint.
    """
    path = np.asarray(path, dtype=float)
    s = arc_length(path)
    if s[-1] < 1e-9:
        return np.repeat(path[:1], 2, axis=0)
    eps = 1e-9
    d1 = np.gradient(path, s, axis=0)            # dq/ds
    d2 = np.gradient(d1, s, axis=0)              # d2q/ds2
    v_ref = max_speed / max(float(np.abs(d1).max()), eps)
    v_lim = np.minimum((max_speed / np.maximum(np.abs(d1), eps)).min(axis=1),
                       np.sqrt(max_accel / np.maximum(np.abs(d2), eps)).min(axis=1))
    sd2_cap = np.maximum(v_lim, 0.01 * v_ref) ** 2
    floor = (0.005 * v_ref) ** 2                 # never stall completely

    def accel_range(i, sd2):
        """Allowed path acceleration at sample i for every joint within +-max_accel."""
        b, c = d1[i], d2[i] * sd2
        ok = np.abs(b) > eps
        if not ok.any():
            return -np.inf, np.inf
        r1, r2 = (-max_accel - c[ok]) / b[ok], (max_accel - c[ok]) / b[ok]
        return float(np.minimum(r1, r2).max()), float(np.maximum(r1, r2).min())

    n = len(s)
    sd2 = np.zeros(n)
    sd2[0] = floor
    for i in range(n - 1):                       # forward: limited acceleration
        _, hi = accel_range(i, sd2[i])
        sd2[i + 1] = min(sd2_cap[i + 1], max(sd2[i] + 2 * (s[i + 1] - s[i]) * hi, floor))
    sd2[-1] = min(sd2[-1], floor)
    for i in range(n - 2, -1, -1):               # backward: limited deceleration
        lo, _ = accel_range(i + 1, sd2[i + 1])
        sd2[i] = min(sd2[i], max(sd2[i + 1] - 2 * (s[i + 1] - s[i]) * lo, floor))
    sd = np.sqrt(np.maximum(sd2, floor))
    t = np.r_[0.0, np.cumsum(2 * np.diff(s) / (sd[:-1] + sd[1:]))]
    grid = np.arange(0.0, t[-1] + dt, dt)
    frames = np.column_stack([np.interp(grid, t, path[:, j]) for j in range(path.shape[1])])
    k = int(round(smooth_time / dt))
    if k > 1:
        # Trailing moving average with the ends held at rest: the first output is
        # exactly the start pose, and the output settles on the goal pose.
        padded = np.vstack([np.repeat(frames[:1], k - 1, axis=0), frames, np.repeat(frames[-1:], k, axis=0)])
        kernel = np.ones(k) / k
        frames = np.column_stack([np.convolve(padded[:, j], kernel, mode='valid')
                                  for j in range(path.shape[1])])
        moving = np.abs(np.diff(frames, axis=0)).max(axis=1) > 1e-7
        frames = frames[:(int(np.nonzero(moving)[0].max()) + 2 if moving.any() else 2)]
    frames[-1] = path[-1]
    return frames


def joint_limits_check(frames, dt, window=0.2):
    """(max joint speed rad/s, max joint acceleration rad/s^2) of sampled frames.

    Acceleration is taken over ``window`` seconds: differencing every frame mostly
    measures the piecewise-linear interpolation of the dense path, not the motion.
    """
    v = np.diff(frames, axis=0) / dt
    k = max(1, int(round(window / dt)))
    if len(v) <= k:
        return float(np.abs(v).max()), 0.0
    smooth = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode='valid'), 0, v)
    return float(np.abs(v).max()), float(np.abs(np.diff(smooth, axis=0) / dt).max())


def _dedupe(waypoints):
    pts = [np.asarray(waypoints[0], dtype=float)]
    for p in waypoints[1:]:
        p = np.asarray(p, dtype=float)
        if np.abs(p - pts[-1]).max() > 1e-9:
            pts.append(p)
    if len(pts) == 1:
        pts.append(pts[0].copy())
    return np.asarray(pts)
