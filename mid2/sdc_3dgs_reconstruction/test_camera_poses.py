"""
Test script: Create Gaussians for point cloud + red Gaussians at each camera pose.
"""

import argparse
import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from utils.sh_utils import RGB2SH


def save_ply(means, colors, scales, filename="visualization.ply"):
    """Save Gaussians as PLY file."""
    num_points = len(means)
    
    # Initialize with identity rotations
    quats = np.zeros((num_points, 4))
    quats[:, 3] = 1.0  # Identity: [0, 0, 0, 1]
    
    # Full opacity
    opacities = np.ones(num_points)
    
    with open(filename, 'wb') as f:
        # Header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")
        f.write(b"property float opacity\n")
        f.write(b"property float f_dc_0\n")
        f.write(b"property float f_dc_1\n")
        f.write(b"property float f_dc_2\n")
        f.write(b"end_header\n")
        
        # Data
        for i in range(num_points):
            # Position
            f.write(struct.pack('f', means[i, 0]))
            f.write(struct.pack('f', means[i, 1]))
            f.write(struct.pack('f', means[i, 2]))
            # Scale (in log space)
            f.write(struct.pack('f', scales[i, 0]))
            f.write(struct.pack('f', scales[i, 1]))
            f.write(struct.pack('f', scales[i, 2]))
            # Rotation
            f.write(struct.pack('f', quats[i, 0]))
            f.write(struct.pack('f', quats[i, 1]))
            f.write(struct.pack('f', quats[i, 2]))
            f.write(struct.pack('f', quats[i, 3]))
            # Opacity
            f.write(struct.pack('f', opacities[i]))
            # Color (RGB in SH DC space)
            f.write(struct.pack('f', colors[i, 0]))
            f.write(struct.pack('f', colors[i, 1]))
            f.write(struct.pack('f', colors[i, 2]))
    
    print(f"Saved: {filename} with {num_points} Gaussians")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_num", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Track root directory. Defaults to mid2dataset/Track{N}")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY path. Defaults to data_dir/visualization.ply")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"mid2dataset/Track{args.track_num}"
    if args.output is None:
        args.output = f"{args.data_dir}/visualization.ply"

    data_dir = args.data_dir

    # Load dataset
    dataset = Dataset(data_dir)
    poses_c2w = dataset.get_poses_torch()  # [N, 4, 4]
    points, colors_rgb = dataset.get_pointcloud()
    
    print(f"Loaded {len(poses_c2w)} camera poses")
    print(f"Loaded {len(points)} point cloud points")
    
    # ========== Point Cloud Gaussians ==========
    pc_means = points.numpy()  # [N_pc, 3]
    pc_colors_rgb = colors_rgb.numpy()  # [N_pc, 3], already in [0, 1]
    
    # Convert RGB to SH DC space
    pc_colors_sh = RGB2SH(torch.from_numpy(pc_colors_rgb).float()).numpy()  # [N_pc, 3]
    
    # All scales = 0.02 in log space
    log_scale = np.log(0.02)
    pc_scales = np.full((len(points), 3), log_scale)  # [N_pc, 3]
    
    # ========== Camera Trajectory Gaussians (red line) ==========
    cam_centers = poses_c2w[:, :3, 3].numpy()  # [N_cam, 3]

    # Interpolate between consecutive camera centers so the trajectory shows up as a line
    samples_per_seg = 50
    seg_pts = []
    for a, b in zip(cam_centers[:-1], cam_centers[1:]):
        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
        seg_pts.append(a[None] * (1.0 - t) + b[None] * t)
    seg_pts.append(cam_centers[-1:].reshape(1, 3))
    cam_means = np.vstack(seg_pts)  # [N_traj, 3]

    # Red color converted to SH DC space
    red_rgb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    red_sh = RGB2SH(red_rgb).numpy()  # [1, 3]
    cam_colors = np.tile(red_sh, (len(cam_means), 1))  # [N_traj, 3]

    # Make the line thick enough to be visible against the point cloud
    line_log_scale = np.log(0.1)
    cam_scales = np.full((len(cam_means), 3), line_log_scale)  # [N_traj, 3]
    
    # ========== Combine ==========
    all_means = np.vstack([pc_means, cam_means])
    all_colors = np.vstack([pc_colors_sh, cam_colors])
    all_scales = np.vstack([pc_scales, cam_scales])
    
    # Save
    ply_path = Path(args.output)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    save_ply(all_means, all_colors, all_scales, str(ply_path))
    
    print(f"\nCombined visualization:")
    print(f"  Point cloud: {len(points)} Gaussians (point colors, 0.02m)")
    print(f"  Camera trajectory: {len(cam_means)} Gaussians (red, 0.1m) from {len(poses_c2w)} poses")
    print(f"  Total: {len(all_means)} Gaussians")
