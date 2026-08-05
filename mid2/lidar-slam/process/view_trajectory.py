import os
import argparse
import numpy as np
import pandas as pd
import open3d as o3d


def load_poses_csv(path):
    df = pd.read_csv(path)
    if df.shape[1] != 17:
        raise ValueError(
            f"Expect 17-col matrix CSV (timestamp + 16 m**), got {df.shape[1]} cols"
        )
    mats = df.iloc[:, 1:].values.reshape(-1, 4, 4)
    return [m for m in mats]


def make_trajectory_lineset(poses, color=(1.0, 1.0, 0.0)):
    pts = np.array([p[:3, 3] for p in poses])
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def make_axes(poses, step=50, scale=1.0):
    geos = []
    for i in range(0, len(poses), step):
        ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        ax.transform(poses[i])
        geos.append(ax)
    return geos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.environ.get(
            "POSES_CSV",
            "mid2dataset/Track3/result/lidar_poses.csv",
        ),
    )
    parser.add_argument("--axis-step", type=int, default=50)
    parser.add_argument("--axis-scale", type=float, default=1.0)
    args = parser.parse_args()

    poses = load_poses_csv(args.csv)
    print(f"[view_trajectory] loaded {len(poses)} poses from {args.csv}")

    traj = make_trajectory_lineset(poses)
    axes = make_axes(poses, step=args.axis_step, scale=args.axis_scale)
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # o3d.io.write_line_set("trajectory.ply", traj)
    # o3d.io.write_triangle_mesh("world.ply", world)
    # print("[view_trajectory] saved trajectory.ply and world.ply for inspection")
    
    o3d.visualization.draw_geometries(
        [traj, world] + axes,
        window_name=f"Trajectory: {os.path.basename(args.csv)}",
        width=1280,
        height=720,
    
    )


if __name__ == "__main__":
    main()
