import os
import rosbag
import random
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os


# or you may use open3d to save pcd
def write_pcd_ascii(filename, points, target_fields):
    """
    Writes point cloud to PCD v0.7 ASCII format with dynamic header.
    :param filename: Output path
    :param points: Numpy array of shape (N, M)
    :param target_fields: Tuple/List of field names (e.g., ("x", "y", "z"))
    """
    num_points = points.shape[0]
    num_fields = len(target_fields)

    # --- Data Validation ---
    if points.shape[1] != num_fields:
        raise ValueError(
            f"Dimension mismatch! Points has {points.shape[1]} columns, "
            f"but target_fields defines {num_fields} ({target_fields})."
        )

    # --- Dynamic Header  ---
    field_names = " ".join(target_fields)
    sizes = " ".join(["4"] * num_fields)   # 4 bytes for float32
    types = " ".join(["F"] * num_fields)   # F for Float
    counts = " ".join(["1"] * num_fields)  # 1 element per dimension

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

    # --- File Writing ---
    with open(filename, 'w') as f:
        f.write(header)
        fmt_str = " ".join(["%.6f"] * num_fields)
        np.savetxt(f, points, fmt=fmt_str)


def main():
    BAG_PATH = "/root/catkin_ws/src/localization/data/Track1/Track1.bag"
    RADAR_TOPIC = "/ars548/radar_front/detections"
    RADAR_OBJECT = "/ars548/radar_front/objects"
    RAW_DIR = "raw_radar_pcd"
    # RADAR_T = "radar_t_list.txt"
    PROCESS_DIR = "/root/catkin_ws/src/localization/data/Track1/radar_pcd_with_intensity"
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESS_DIR, exist_ok=True)

    pcd_names = []
    bag = rosbag.Bag(BAG_PATH)
    for topic, msg, t in bag.read_messages(topics=[RADAR_TOPIC]):
        ts_nsec_str = str(msg.header.stamp.to_nsec())
        ts_sec = msg.header.stamp.to_sec()

        target_fields = ("x", "y", "z", "RCS") # read supplement in ppt to check all field, or you already know them during HW2
        pc_data = list(pc2.read_points(msg, field_names=target_fields, skip_nans=True))
        if not pc_data:
            continue

        # transform into np
        current_pts = np.array(pc_data)[:, :4]

        pcd_names.append(ts_nsec_str)

        #  Raw PCD out — RCS is written as "intensity" so PCL can load it via PointXYZI
        processed_pts = current_pts

        """
        Blank for preprocessing radar point cloud, make sure processed_pts is np array.
        """
        pcd_field_names = ["x", "y", "z", "intensity"]
        write_pcd_ascii(os.path.join(PROCESS_DIR, f"{ts_nsec_str}.pcd"), processed_pts, pcd_field_names)
    bag.close()

    # # already provided
    # # # --- radar_t_list.txt ---
    # RADAR_T = "/root/catkin_ws/src/localization/data/Track1/radar_t_list.txt"
    # with open(RADAR_T, 'w') as f:
    #     for name in pcd_names:
    #         f.write(f"{name}\n")


if __name__ == "__main__":
    main()
