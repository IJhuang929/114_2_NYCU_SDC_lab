import csv
import math
import argparse
import numpy as np


def txt_to_poses(input_txt):
    """Read result.txt and return list of (x, y, yaw) tuples."""
    poses = []
    with open(input_txt, 'r') as f:
        for line in f:
            vals = list(map(float, line.split()))
            x = vals[3]
            y = vals[7]
            yaw = math.atan2(vals[4], vals[0])
            poses.append((x, y, yaw))
    return poses


def evaluate(poses):
    """Evaluate localization quality based on trajectory smoothness."""
    n = len(poses)
    if n < 2:
        print("Not enough frames to evaluate.")
        return

    # Frame-to-frame displacement
    displacements = []
    yaw_changes = []
    for i in range(1, n):
        dx = poses[i][0] - poses[i-1][0]
        dy = poses[i][1] - poses[i-1][1]
        dist = math.sqrt(dx*dx + dy*dy)
        displacements.append(dist)

        dyaw = abs(poses[i][2] - poses[i-1][2])
        if dyaw > math.pi:
            dyaw = 2 * math.pi - dyaw
        yaw_changes.append(math.degrees(dyaw))

    displacements = np.array(displacements)
    yaw_changes = np.array(yaw_changes)

    print(f"=== Localization Quality Report ===")
    print(f"Total frames: {n}")
    print()
    print(f"--- Frame-to-frame displacement (m) ---")
    print(f"  Mean:   {displacements.mean():.3f}")
    print(f"  Std:    {displacements.std():.3f}")
    print(f"  Max:    {displacements.max():.3f}")
    print(f"  Median: {np.median(displacements):.3f}")
    print()
    print(f"--- Frame-to-frame yaw change (deg) ---")
    print(f"  Mean:   {yaw_changes.mean():.3f}")
    print(f"  Std:    {yaw_changes.std():.3f}")
    print(f"  Max:    {yaw_changes.max():.3f}")
    print()

    # Flag abnormal frames (large jumps)
    jump_threshold = displacements.mean() + 3 * displacements.std()
    yaw_threshold = yaw_changes.mean() + 3 * yaw_changes.std()
    bad_frames = []
    for i in range(len(displacements)):
        if displacements[i] > jump_threshold or yaw_changes[i] > yaw_threshold:
            bad_frames.append(i + 1)

    if bad_frames:
        print(f"--- Abnormal frames (jump > {jump_threshold:.2f}m or yaw > {yaw_threshold:.1f} deg) ---")
        for f in bad_frames:
            print(f"  Frame {f}: displacement={displacements[f-1]:.3f}m, yaw_change={yaw_changes[f-1]:.2f} deg")
    else:
        print("No abnormal frames detected.")

    print()
    print(f"--- Trajectory ---")
    print(f"  Start: x={poses[0][0]:.2f}, y={poses[0][1]:.2f}")
    print(f"  End:   x={poses[-1][0]:.2f}, y={poses[-1][1]:.2f}")
    total_dist = displacements.sum()
    print(f"  Total distance: {total_dist:.2f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate localization quality")
    parser.add_argument("input", help="result.txt file path")
    args = parser.parse_args()

    poses = txt_to_poses(args.input)
    evaluate(poses)

# python3 /root/catkin_ws/src/localization/processing_codes/eval_score.py /root/catkin_ws/src/localization/result/result.txt
# displacement std 越小越好（軌跡越平滑）
# abnormal frames 越少越好
# max jump 越小越好，理想 < 1m
# 如果助教有提供 ground truth，可以再加上和 GT 的 RMSE 計算
