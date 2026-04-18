#!/usr/bin/env python3
"""
Radar preprocessor + IMU extractor.

Usage:
    python3 radar_preprocessor.py params.yaml              # radar PCD only
    python3 radar_preprocessor.py params.yaml --extract-imu # radar PCD + IMU aligned txt
    python3 radar_preprocessor.py params.yaml --imu-only    # IMU only (skip radar PCD)

Reads all paths and parameters from params.yaml.
"""
import os
import sys
import glob
import yaml
import numpy as np
from collections import deque
from math import atan2, asin, pi
from bisect import bisect_left
import open3d as o3d
import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None
    print("[WARN] scipy not found, SOR filter will be skipped")

# ============================================================
#  Point cloud filters
# ============================================================

def range_filter(points, r_min, r_max):
    """距離過濾"""
    dists = np.linalg.norm(points[:, :3], axis=1)
    mask = (dists >= r_min) & (dists <= r_max)
    return points[mask]

def pass_through_filter(points, z_min, z_max):
    """Z 範圍過濾"""
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]

def statistical_outlier_removal(points, mean_k=10, std_dev_mul=2):
    """統計離群值移除 (SOR)"""
    if KDTree is None or len(points) < mean_k:
        return points
    tree = KDTree(points[:, :3])
    dists, _ = tree.query(points[:, :3], k=mean_k + 1)
    avg_dists = np.mean(dists[:, 1:], axis=1)
    threshold = np.mean(avg_dists) + std_dev_mul * np.std(avg_dists)
    return points[avg_dists < threshold]

def voxel_grid_filter(points, leaf_size):
    """體素下採樣"""
    if len(points) == 0 or leaf_size <= 0:
        return points
    idx = np.floor(points[:, :3] / leaf_size).astype(np.int64)
    keys = idx[:, 0] * 1000003 + idx[:, 1] * 1009 + idx[:, 2]
    _, indices = np.unique(keys, return_index=True)
    return points[indices]

def intensity_range_compensation(points):
    """
    雷達強度距離補償 (Intensity Range Compensation)
    Pn = log(R^4 * Pr) = 4*log(R) + log(Pr)

    Input:  (N, 4) -> [x, y, z, rcs_dB]
    Output: (N, 4) -> [x, y, z, compensated_intensity]

    R  = sqrt(x^2 + y^2 + z^2)  (距離)
    Pr = 10^(rcs_dB / 10)        (dB 轉線性功率)
    """
    if len(points) == 0:
        return points
    xyz = points[:, :3]
    rcs = points[:, 3]
    R = np.maximum(np.linalg.norm(xyz, axis=1), 0.1)
    Pr = np.maximum(np.power(10.0, rcs / 10.0), 1e-6)
    Pn = 4.0 * np.log(R) + np.log(Pr)
    result = points.copy()
    result[:, 3] = Pn
    return result

# ============================================================
#  PCD I/O
# ============================================================

def read_pcd(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_idx = next(i for i, l in enumerate(lines) if l.startswith('DATA')) + 1
    rows = [line.strip().split() for line in lines[data_idx:] if line.strip()]
    if not rows:
        return np.empty((0, 4))
    arr = np.array(rows, dtype=np.float32)
    if arr.shape[1] >= 5:
        return arr[:, [0, 1, 2, 4]]   # x, y, z, rcs (skip range col)
    elif arr.shape[1] >= 4:
        return arr[:, :4]              # x, y, z, intensity
    else:
        return np.column_stack([arr[:, :3], np.zeros(len(arr))])  # xyz only

def write_pcd(filepath, points):
    n = len(points)
    header = (
        f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        f"TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA ascii\n"
    )
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")

# ============================================================
#  IMU utilities
# ============================================================

def normalize_angle(angle):
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle

def quaternion_to_rpy(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (pi / 2.0) if abs(sinp) >= 1 else asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def find_closest_index(sorted_list, target):
    pos = bisect_left(sorted_list, target)
    if pos == 0:
        return 0
    if pos == len(sorted_list):
        return len(sorted_list) - 1
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    return pos if abs(after - target) < abs(target - before) else pos - 1

def detect_imu_topic(bag):
    """Auto-detect the first sensor_msgs/Imu topic in the bag."""
    topic_info = bag.get_type_and_topic_info().topics
    candidates = [name for name, info in topic_info.items()
                  if info.msg_type == "sensor_msgs/Imu"]
    if not candidates:
        return None
    for keyword in ["/imu/data", "/imu", "imu/data", "imu"]:
        for c in candidates:
            if keyword in c:
                return c
    return candidates[0]

def extract_imu_table(bag, imu_topic):
    """
    Extract full-rate IMU data from the bag.
    Returns list of dicts: {t_ns, roll, pitch, yaw, ax, ay, source}
    Handles both orientation-based and gyro-integration fallback.
    ax, ay = body-frame linear acceleration (from sensor_msgs/Imu).
    """
    imu_table = []
    prev_t = None
    integrated_yaw = 0.0
    prev_orientation_yaw = None

    for _, msg, _ in bag.read_messages(topics=[imu_topic]):
        t_ns = msg.header.stamp.to_nsec()
        qx, qy, qz, qw = (msg.orientation.x, msg.orientation.y,
                            msg.orientation.z, msg.orientation.w)

        # Body-frame linear acceleration (always available)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y

        has_valid_orientation = not (
            abs(qx) < 1e-12 and abs(qy) < 1e-12 and
            abs(qz) < 1e-12 and abs(qw) < 1e-12
        )

        if has_valid_orientation:
            roll, pitch, yaw = quaternion_to_rpy(qx, qy, qz, qw)
            # Unwrap yaw to avoid +-pi jumps
            if prev_orientation_yaw is not None:
                dyaw = normalize_angle(yaw - prev_orientation_yaw)
                yaw = imu_table[-1]["yaw"] + dyaw if imu_table else yaw
            prev_orientation_yaw = quaternion_to_rpy(qx, qy, qz, qw)[2]
            imu_table.append({"t_ns": t_ns, "roll": roll, "pitch": pitch,
                              "yaw": yaw, "ax": ax, "ay": ay,
                              "source": "orientation"})
        else:
            # Fallback: integrate gyro z
            dt = 0.0 if prev_t is None else (t_ns - prev_t) * 1e-9
            wz = msg.angular_velocity.z
            integrated_yaw = normalize_angle(integrated_yaw + wz * dt)
            imu_table.append({"t_ns": t_ns, "roll": 0.0, "pitch": 0.0,
                              "yaw": integrated_yaw, "ax": ax, "ay": ay,
                              "source": "gyro_integrated"})

        prev_t = t_ns

    return imu_table

def build_imu_aligned_txt(radar_timestamps_ns, imu_table, output_txt):
    """
    For each radar timestamp, find the nearest IMU sample and compute delta_yaw.
    Output format (space-separated):
        timestamp roll pitch yaw delta_yaw ax ay source
    ax, ay are body-frame linear acceleration from IMU.
    """
    if not imu_table:
        raise RuntimeError("IMU table is empty!")

    imu_times = [d["t_ns"] for d in imu_table]
    prev_yaw = None

    with open(output_txt, "w") as f:
        f.write("timestamp roll pitch yaw delta_yaw ax ay source\n")
        for radar_t_ns in radar_timestamps_ns:
            idx = find_closest_index(imu_times, radar_t_ns)
            d = imu_table[idx]
            yaw = d["yaw"]
            delta_yaw = 0.0 if prev_yaw is None else normalize_angle(yaw - prev_yaw)
            prev_yaw = yaw
            f.write(f"{radar_t_ns} {d['roll']:.9f} {d['pitch']:.9f} "
                    f"{yaw:.9f} {delta_yaw:.9f} "
                    f"{d['ax']:.9f} {d['ay']:.9f} {d['source']}\n")

    print(f"[OK] imu_aligned.txt saved: {output_txt}  ({len(radar_timestamps_ns)} entries)")

# ============================================================
#  Main pipeline
# ============================================================

def resolve(params, key, track):
    """Resolve a params.yaml value, replacing $(arg track)."""
    val = params.get(key, "")
    return val.replace("$(arg track)", track) if val else ""



def process_radar_pcd(params, track):
    """Stage 1: filter + stack + SOR + voxel on raw radar PCD files."""
    raw_folder = resolve(params, 'radar_pcd_folder', track)
    output_dir = resolve(params, 'preprocessed_radar_pcd_folder', track)

    z_min       = float(params.get('radar_z_min', -10.0))
    z_max       = float(params.get('radar_z_max', 30.0))
    r_min       = float(params.get('range_min', 1.5))
    r_max       = float(params.get('range_max', 70.0))
    n_frames    = int(params.get('accumulate_frames', 3))
    v_size      = float(params.get('accumulate_voxel_size', 0.5))
    mean_k      = int(params.get('radar_sol_mean_k', 20))
    std_dev_mul = float(params.get('radar_sol_std_dev_mul', 1.5))
    rcs_min          = float(params.get('radar_rcs_min', 2.0))       

    os.makedirs(output_dir, exist_ok=True)
    pcd_files = sorted(glob.glob(os.path.join(raw_folder, '*.pcd')))
    frame_buffer = deque(maxlen=n_frames)

    print(f"--- Radar PCD preprocessing ({track}) - [No Doppler Mode] ---")
    print(f"    Z[{z_min}~{z_max}], Range[{r_min}~{r_max}], Voxel: {v_size}")
    print(f"    Filtering: RCS_min={rcs_min} (Used to suppress ghosts)")
    for pcd_path in pcd_files:
        basename = os.path.basename(pcd_path)
        
        # 這裡假設你的 read_pcd 讀取後包含 'x','y','z','intensity' 欄位
        points = read_pcd(pcd_path) 
        if len(points) == 0:
            continue
        
        # 1. RCS Filter: 移除強度過低的點 (Ghost points/Noise)
        if 'intensity' in points.dtype.names:
            points = points[points['intensity'] > rcs_min]

        if len(points) == 0:
            continue

        # 強度距離補償: Pn = 4*log(R) + log(Pr)
        points = intensity_range_compensation(points)

        # 空間過濾
        points = pass_through_filter(points, z_min, z_max)
        points = range_filter(points, r_min, r_max)

        # 提取 XYZ 用於後續堆疊與 PCL 過濾
        # 注意：如果 read_pcd 回傳的是自定義結構，這裡需要提取 XYZ
        xyz = np.vstack((points['x'], points['y'], points['z'])).T

        frame_buffer.append(xyz)
        stacked = np.vstack(list(frame_buffer))

        # PCL 風格的過濾
        filtered = statistical_outlier_removal(stacked, mean_k, std_dev_mul)
        final = voxel_grid_filter(filtered, v_size)

        write_pcd(os.path.join(output_dir, basename), final)

    print(f"[OK] Radar PCD done -> {output_dir}")






def extract_imu(params, track):
    """Stage 2: open the rosbag, extract IMU, align to radar timestamps."""
    import rosbag  # only import when needed (requires ROS environment)

    bag_path    = resolve(params, 'bag_path', track)
    t_list_path = resolve(params, 'radar_t_list', track)
    imu_out     = resolve(params, 'imu_aligned_path', track)

    # Validate inputs
    if not bag_path or not os.path.isfile(bag_path):
        print(f"[ERROR] bag file not found: {bag_path}")
        return
    if not t_list_path or not os.path.isfile(t_list_path):
        print(f"[ERROR] radar_t_list not found: {t_list_path}")
        return

    # Read radar timestamps from radar_t_list.txt
    with open(t_list_path, 'r') as f:
        radar_timestamps_ns = [int(line.strip()) for line in f if line.strip()]
    print(f"[INFO] Radar timestamps: {len(radar_timestamps_ns)} frames from {t_list_path}")

    # Open bag and detect IMU topic
    print(f"[INFO] Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path)

    imu_topic = detect_imu_topic(bag)
    if imu_topic is None:
        bag.close()
        print("[ERROR] No sensor_msgs/Imu topic found in bag!")
        return
    print(f"[INFO] Detected IMU topic: {imu_topic}")

    # Extract high-rate IMU table
    imu_table = extract_imu_table(bag, imu_topic)
    print(f"[INFO] IMU messages extracted: {len(imu_table)}")
    bag.close()

    if not imu_table:
        print("[ERROR] IMU table is empty, cannot produce imu_aligned.txt")
        return

    # Align to radar timestamps and save
    os.makedirs(os.path.dirname(imu_out) or ".", exist_ok=True)
    build_imu_aligned_txt(radar_timestamps_ns, imu_table, imu_out)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 radar_preprocessor.py params.yaml [--extract-imu] [--imu-only]")
        print()
        print("  (default)        Radar PCD preprocessing only")
        print("  --extract-imu    Radar PCD + extract IMU from bag -> imu_aligned.txt")
        print("  --imu-only       Only extract IMU (skip radar PCD preprocessing)")
        return

    with open(sys.argv[1], 'r') as f:
        params = yaml.safe_load(f)

    track    = params.get('track', 'Track1')
    do_imu   = '--extract-imu' in sys.argv
    imu_only = '--imu-only' in sys.argv

    # Stage 1: Radar PCD preprocessing
    if not imu_only:
        process_radar_pcd(params, track)

    # Stage 2: IMU extraction from bag
    if do_imu or imu_only:
        extract_imu(params, track)
    else:
        print("[INFO] Skipping IMU extraction (use --extract-imu or --imu-only to enable)")

if __name__ == '__main__':
    main()
