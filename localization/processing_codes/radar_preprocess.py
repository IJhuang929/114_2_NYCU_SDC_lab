import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from collections import deque


def write_pcd_ascii(filename, points, target_fields):
    """
    Writes point cloud to PCD v0.7 ASCII format with dynamic header.
    """
    num_points = points.shape[0]
    num_fields = len(target_fields)

    if points.shape[1] != num_fields:
        raise ValueError(
            f"Dimension mismatch! Points has {points.shape[1]} columns, "
            f"but target_fields defines {num_fields} ({target_fields})."
        )

    field_names = " ".join(target_fields)
    sizes = " ".join(["4"] * num_fields)
    types = " ".join(["F"] * num_fields)
    counts = " ".join(["1"] * num_fields)

    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"FIELDS {field_names}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {num_points}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_points}\n"
        f"DATA ascii\n"
    )

    with open(filename, 'w') as f:
        f.write(header)
        fmt_str = " ".join(["%.6f"] * num_fields)
        np.savetxt(f, points, fmt=fmt_str)


def main():
    BAG_PATH = "/root/catkin_ws/src/localization/data/Track2/Track2.bag"
    RADAR_TOPIC = "/ars548/radar_front/detections"
    STACK_SIZE = 3
    OUTPUT_DIR = "/root/catkin_ws/src/localization/data/Track2/stack_3_frame_radar_pcd"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 先讀取所有 radar 幀
    frames = []
    bag = rosbag.Bag(BAG_PATH)
    for topic, msg, t in bag.read_messages(topics=[RADAR_TOPIC]):
        ts_nsec_str = str(msg.header.stamp.to_nsec())

        target_fields = ("x", "y", "z")
        pc_data = list(pc2.read_points(msg, field_names=target_fields, skip_nans=True))
        if not pc_data:
            continue

        pts = np.array(pc_data)[:, :3]  # 只取 x, y, z
        frames.append((ts_nsec_str, pts))
    bag.close()

    print(f"Total radar frames: {len(frames)}")

    # 疊加 3 幀輸出
    buffer = deque(maxlen=STACK_SIZE)
    for i, (ts_name, pts) in enumerate(frames):
        buffer.append(pts)

        # 把 buffer 內所有幀疊在一起
        stacked = np.vstack(list(buffer))

        pcd_fields = ["x", "y", "z"]
        output_path = os.path.join(OUTPUT_DIR, f"{ts_name}.pcd")
        write_pcd_ascii(output_path, stacked, pcd_fields)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)}, stacked points: {stacked.shape[0]}")

    print(f"Done. Output {len(frames)} stacked PCD files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
