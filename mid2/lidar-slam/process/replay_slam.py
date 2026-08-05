"""
Offline SLAM replay: renders frame-by-frame using Open3D OffscreenRenderer
(no display required), saves PNGs, then encodes to mp4 via ffmpeg.

Usage:
    TARGET=Track1 python process/replay_slam.py
    TARGET=Track1 python process/replay_slam.py --fps 20 --step 2 --voxel 0.4
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import pandas as pd
from tqdm import tqdm

TARGET       = os.environ.get("TARGET", "Track1")
DATASET_ROOT = os.environ.get("MID2_DATASET_ROOT", "mid2dataset")
W, H         = 640, 480


def load_poses_csv(path):
    df = pd.read_csv(path)
    timestamps = df.iloc[:, 0].astype(np.int64).values
    poses      = df.iloc[:, 1:].values.reshape(-1, 4, 4)
    return poses, timestamps


def _mat_points(color, size=2.0):
    m = rendering.MaterialRecord()
    m.shader     = "defaultUnlit"
    m.base_color = [*color, 1.0]
    m.point_size = size
    return m


def _mat_line(color, width=2.0):
    m = rendering.MaterialRecord()
    m.shader     = "unlitLine"
    m.base_color = [*color, 1.0]
    m.line_width = width
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",   type=int,   default=15,  help="Output video FPS")
    parser.add_argument("--step",  type=int,   default=1,   help="Render every N frames (>1 speeds up encoding)")
    parser.add_argument("--voxel", type=float, default=0.5, help="Map downsampling voxel size (m)")
    parser.add_argument("--cam-dist",  type=float, default=15.0, help="Camera distance behind robot (m)")
    parser.add_argument("--cam-height",type=float, default=25.0, help="Camera height above robot (m)")
    args = parser.parse_args()

    pcd_dir    = os.path.join(DATASET_ROOT, TARGET, "data", "raw_pcd")
    poses_csv  = os.path.join(DATASET_ROOT, TARGET, "result", "lidar_poses.csv")
    output_mp4 = os.path.join(DATASET_ROOT, TARGET, "result", f"{TARGET}_replay.mp4")
    frames_dir = f"/tmp/slam_replay_{TARGET}"

    if not os.path.exists(poses_csv):
        raise FileNotFoundError(f"poses CSV not found: {poses_csv}")

    os.makedirs(frames_dir, exist_ok=True)

    poses, pose_ts = load_poses_csv(poses_csv)
    pcd_files      = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    print(f"[replay] target={TARGET}, frames={len(pcd_files)}, poses={len(poses)}")

    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0.05, 0.05, 0.05, 1.0])

    mat_map  = _mat_points([0.65, 0.65, 0.65], size=2.0)
    mat_scan = _mat_points([0.0,  1.0,  0.0 ], size=3.0)
    mat_traj = _mat_line  ([1.0,  1.0,  0.0 ], width=2.0)

    global_pts  = np.zeros((0, 3), dtype=np.float64)
    traj_pts    = []
    saved_frames = 0

    for i, pcd_file in enumerate(tqdm(pcd_files, desc="Rendering")):
        if i % args.step != 0:
            continue

        ts       = int(os.path.splitext(os.path.basename(pcd_file))[0])
        pose_idx = int(np.argmin(np.abs(pose_ts - ts)))
        pose     = poses[pose_idx]

        pcd = o3d.io.read_point_cloud(pcd_file)
        pts = np.asarray(pcd.points)[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts_world = (pose[:3, :3] @ pts.T).T + pose[:3, 3]

        global_pts = np.concatenate([global_pts, pts_world], axis=0)
        if len(global_pts) > 300_000:
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(global_pts)
            tmp = tmp.voxel_down_sample(args.voxel)
            global_pts = np.asarray(tmp.points, dtype=np.float64)

        traj_pts.append(pose[:3, 3].copy())

        # --- build geometries ---
        map_geo  = o3d.geometry.PointCloud()
        map_geo.points = o3d.utility.Vector3dVector(global_pts)

        scan_geo = o3d.geometry.PointCloud()
        scan_geo.points = o3d.utility.Vector3dVector(pts_world)

        renderer.scene.remove_geometry("map")
        renderer.scene.remove_geometry("scan")
        renderer.scene.remove_geometry("traj")

        renderer.scene.add_geometry("map",  map_geo,  mat_map)
        renderer.scene.add_geometry("scan", scan_geo, mat_scan)

        if len(traj_pts) > 1:
            t_arr  = np.array(traj_pts)
            lines  = [[j, j + 1] for j in range(len(t_arr) - 1)]
            ls     = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(t_arr)
            ls.lines  = o3d.utility.Vector2iVector(lines)
            renderer.scene.add_geometry("traj", ls, mat_traj)

        # --- camera: follow robot from above-behind ---
        center = pose[:3, 3]
        fwd    = pose[:3, 0]
        eye    = center - fwd * args.cam_dist + np.array([0.0, 0.0, args.cam_height])
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), [0.0, 0.0, 1.0])

        img = renderer.render_to_image()
        o3d.io.write_image(
            os.path.join(frames_dir, f"frame_{saved_frames:05d}.png"), img
        )
        saved_frames += 1

    print(f"\n[replay] saved {saved_frames} frames → encoding mp4...")

    if shutil.which("ffmpeg") is None:
        print("[replay] ffmpeg not found. Install with: sudo apt install ffmpeg")
        print(f"[replay] frames are in: {frames_dir}")
        return

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(frames_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            output_mp4,
        ],
        check=True,
    )

    shutil.rmtree(frames_dir)
    print(f"[replay] done → {output_mp4}")


if __name__ == "__main__":
    main()
