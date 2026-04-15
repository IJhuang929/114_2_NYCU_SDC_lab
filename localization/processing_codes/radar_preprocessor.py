#!/usr/bin/env python3
"""
Radar Preprocessor (CLI mode)

Usage:
    python3 radar_preprocessor.py <params.yaml>

Reads radar PCD files from the folder specified in params.yaml,
applies filtering / ego-velocity RANSAC / multi-frame accumulation,
and saves per-frame accumulated PCD files + ego_velocity.csv.
"""
import os
import sys
import glob
import yaml
import csv
import numpy as np
from collections import deque


# ============================================================
#  PCD I/O
# ============================================================
def read_pcd(filepath):
    """Read ASCII PCD file, return Nx5 array [x, y, z, RangeRate, RCS]."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('DATA'):
            data_idx = i + 1
            break

    points = []
    for line in lines[data_idx:]:
        vals = line.strip().split()
        if len(vals) >= 5:
            points.append([float(v) for v in vals[:5]])

    if len(points) == 0:
        return np.empty((0, 5))
    return np.array(points)


def write_pcd_xyzi(filepath, points):
    """
    Write ASCII PCD with fields: x y z intensity
    points: Nx4 [x, y, z, intensity(rcs)]
    """
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")


# ============================================================
#  Step 1: Single-frame filtering
# ============================================================
def filter_single_frame(points, rcs_min, z_min, z_max, range_min, range_max):
    """
    points: Nx5 [x, y, z, RangeRate, RCS]
    Returns filtered Nx5.
    """
    if len(points) == 0:
        return points

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rcs = points[:, 4]
    ranges = np.sqrt(x**2 + y**2 + z**2)

    mask = (
        (rcs > rcs_min) &
        (z > z_min) & (z < z_max) &
        (ranges > range_min) & (ranges < range_max)
    )
    return points[mask]


# ============================================================
#  Step 2: Ego velocity estimation (RANSAC)
# ============================================================
def estimate_ego_velocity(points, doppler_threshold, n_iter=200, n_sample=3):
    """
    RANSAC estimation of ego velocity from radar doppler.
    points: Nx5 [x, y, z, RangeRate, RCS]
    Returns: v_ego (3,), inlier_mask (N,)
    """
    if len(points) < n_sample:
        return np.zeros(3), np.ones(len(points), dtype=bool)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    doppler = points[:, 3]

    # Azimuth and elevation
    az = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    el = np.arctan2(z, r_xy)

    # Direction matrix: v_r = -D @ v_ego
    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)
    D = np.column_stack([cos_az * cos_el, sin_az * cos_el, sin_el])

    best_inliers = 0
    best_v_ego = np.zeros(3)
    best_mask = np.zeros(len(points), dtype=bool)

    N = len(points)
    for _ in range(n_iter):
        idx = np.random.choice(N, n_sample, replace=False)
        D_s = D[idx]
        doppler_s = doppler[idx]

        try:
            v_ego, _, _, _ = np.linalg.lstsq(D_s, -doppler_s, rcond=None)
        except np.linalg.LinAlgError:
            continue

        doppler_pred = -D @ v_ego
        residual = np.abs(doppler - doppler_pred)
        inlier_mask = residual < doppler_threshold

        n_inliers = np.sum(inlier_mask)
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_mask = inlier_mask
            best_v_ego = v_ego

    # Refit with all inliers
    if best_inliers >= n_sample:
        D_in = D[best_mask]
        doppler_in = doppler[best_mask]
        try:
            best_v_ego, _, _, _ = np.linalg.lstsq(D_in, -doppler_in, rcond=None)
        except np.linalg.LinAlgError:
            pass

    return best_v_ego, best_mask


# ============================================================
#  Step 3: Multi-frame accumulation with ego-motion compensation
# ============================================================
def accumulate_frames(frame_buffer, voxel_size):
    """
    frame_buffer: list of (points_xyz_rcs Nx4, dt_to_latest, v_ego (3,))
    Compensate each old frame to latest frame coordinate, merge, voxel downsample.
    Returns Nx4 [x, y, z, rcs].
    """
    if len(frame_buffer) == 0:
        return np.empty((0, 4))

    all_points = []
    for pts, dt, v_ego in frame_buffer:
        if len(pts) == 0:
            continue
        compensated = pts.copy()
        compensated[:, 0] += v_ego[0] * dt
        compensated[:, 1] += v_ego[1] * dt
        compensated[:, 2] += v_ego[2] * dt
        all_points.append(compensated)

    if len(all_points) == 0:
        return np.empty((0, 4))

    merged = np.vstack(all_points)
    return voxel_downsample(merged, voxel_size)


def voxel_downsample(points, voxel_size):
    """
    Voxel grid downsample for Nx4 [x, y, z, rcs].
    Within each voxel, average xyz and rcs.
    """
    if len(points) == 0 or voxel_size <= 0:
        return points

    voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int64)

    # Hash-based grouping
    keys = voxel_idx[:, 0] * 1000003 + voxel_idx[:, 1] * 1009 + voxel_idx[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    result = np.zeros((len(unique_keys), 4))
    counts = np.zeros(len(unique_keys))
    np.add.at(result, inverse, points)
    np.add.at(counts, inverse, 1)
    result /= counts[:, np.newaxis]

    return result


# ============================================================
#  Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <params.yaml>")
        sys.exit(1)

    # --- Load params ---
    with open(sys.argv[1], 'r') as f:
        params = yaml.safe_load(f)

    # Resolve $(arg track) substitution (ROS launch syntax not available in CLI)
    track = params.get('track', 'Track1')
    for key, val in params.items():
        if isinstance(val, str) and '$(arg track)' in val:
            params[key] = val.replace('$(arg track)', track)

    radar_pcd_folder  = params.get('radar_pcd_folder',
                        '/root/catkin_ws/src/localization/data/Track1/raw_radar_pcd/')
    rcs_min           = float(params.get('rcs_min_threshold', -10.0))
    z_min             = float(params.get('radar_z_min', -1.5))
    z_max             = float(params.get('radar_z_max', 3.0))
    range_min         = float(params.get('range_min', 1.5))
    range_max         = float(params.get('range_max', 80.0))
    doppler_threshold = float(params.get('doppler_inlier_threshold', 0.15))
    n_accumulate      = int(params.get('accumulate_frames', 10))
    voxel_size        = float(params.get('accumulate_voxel_size', 0.3))

    # Output folder: next to input folder
    output_dir = os.path.join(os.path.dirname(radar_pcd_folder.rstrip('/')),
                              'preprocessed_radar_pcd')
    os.makedirs(output_dir, exist_ok=True)

    # --- Load all PCD files sorted by timestamp ---
    pcd_files = sorted(glob.glob(os.path.join(radar_pcd_folder, '*.pcd')))
    if len(pcd_files) == 0:
        print(f"[ERROR] No PCD files found in {radar_pcd_folder}")
        sys.exit(1)

    print(f"Found {len(pcd_files)} radar PCD files in {radar_pcd_folder}")
    print(f"Output dir: {output_dir}")

    # Sliding window: (points_xyz_rcs, dt, v_ego, timestamp_sec)
    frame_buffer = deque(maxlen=n_accumulate)

    # Ego velocity log
    ego_vel_csv_path = os.path.join(output_dir, 'ego_velocity.csv')
    ego_vel_file = open(ego_vel_csv_path, 'w', newline='')
    ego_vel_writer = csv.writer(ego_vel_file)
    ego_vel_writer.writerow(['timestamp_nsec', 'vx', 'vy', 'vz'])

    for i, pcd_path in enumerate(pcd_files):
        basename = os.path.splitext(os.path.basename(pcd_path))[0]
        try:
            ts_nsec = int(basename)
        except ValueError:
            print(f"  [WARN] Skipping non-timestamp file: {basename}")
            continue

        current_ts_sec = ts_nsec / 1e9

        # --- Read PCD ---
        raw_points = read_pcd(pcd_path)
        if len(raw_points) == 0:
            continue

        # --- Step 1: Single-frame filtering ---
        filtered = filter_single_frame(raw_points, rcs_min, z_min, z_max,
                                       range_min, range_max)

        # --- Step 2: Ego velocity RANSAC ---
        v_ego, inlier_mask = estimate_ego_velocity(filtered, doppler_threshold)
        static_points = filtered[inlier_mask]

        # Log ego velocity
        ego_vel_writer.writerow([basename, f"{v_ego[0]:.6f}",
                                 f"{v_ego[1]:.6f}", f"{v_ego[2]:.6f}"])

        print(f"  [{i+1:4d}/{len(pcd_files)}] raw={len(raw_points):4d}  "
              f"filtered={len(filtered):4d}  static={len(static_points):4d}  "
              f"v_ego=[{v_ego[0]:+.2f}, {v_ego[1]:+.2f}, {v_ego[2]:+.2f}]")

        # Prepare [x, y, z, rcs] for accumulation
        if len(static_points) > 0:
            pts_xyz_rcs = np.column_stack([
                static_points[:, :3],
                static_points[:, 4],   # RCS -> intensity
            ])
        else:
            pts_xyz_rcs = np.empty((0, 4))

        # --- Step 3: Multi-frame accumulation ---
        new_buffer = deque(maxlen=n_accumulate)
        for old_pts, _, old_v, old_ts in frame_buffer:
            new_dt = current_ts_sec - old_ts
            new_buffer.append((old_pts, new_dt, old_v, old_ts))
        new_buffer.append((pts_xyz_rcs, 0.0, v_ego, current_ts_sec))
        frame_buffer = new_buffer

        # Accumulate and downsample
        accum_input = [(pts, dt, vel) for pts, dt, vel, _ in frame_buffer]
        accumulated = accumulate_frames(accum_input, voxel_size)

        # Save accumulated PCD
        if len(accumulated) > 0:
            out_path = os.path.join(output_dir, f"{basename}.pcd")
            write_pcd_xyzi(out_path, accumulated)

    ego_vel_file.close()
    print(f"\nDone. Saved {len(pcd_files)} preprocessed PCD files to {output_dir}")
    print(f"Ego velocity log: {ego_vel_csv_path}")


if __name__ == '__main__':
    main()
