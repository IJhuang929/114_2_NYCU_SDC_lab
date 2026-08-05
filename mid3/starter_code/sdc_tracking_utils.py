from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_mask_file(mask_file: str | Path) -> np.ndarray:
    mask_file = Path(mask_file)
    with mask_file.open("r") as f:
        values = [int(line.strip()) for line in f if line.strip()]
    return np.asarray(values, dtype=np.int32)


def read_bin(bin_path: str | Path) -> np.ndarray:
    bin_path = Path(bin_path)
    radar_pc = np.fromfile(bin_path, dtype=np.float32)

    if radar_pc.size % 5 != 0:
        raise ValueError(f"Radar binary size is not divisible by 5: {bin_path}")

    return radar_pc.reshape(-1, 5)


def list_frame_names(mask_dir: str | Path) -> list[str]:
    mask_dir = Path(mask_dir)
    names = [p.name for p in mask_dir.iterdir() if p.is_file()]
    return sorted(names)


def radar_path_from_mask_name(radar_dir: str | Path, mask_name: str) -> Path:
    radar_dir = Path(radar_dir)
    stem = Path(mask_name).stem

    candidates = [
        radar_dir / f"{stem}.bin",
        radar_dir / f"{mask_name[:4]}.bin",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find radar binary for mask file {mask_name}. Tried: {candidates}"
    )


def load_radar_and_mask_data(
    mask_dir: str | Path,
    radar_dir: str | Path,
    mask_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    mask_dir = Path(mask_dir)
    radar_dir = Path(radar_dir)

    mask_path = mask_dir / mask_name
    radar_path = radar_path_from_mask_name(radar_dir, mask_name)

    mask = read_mask_file(mask_path)
    radar_pc = read_bin(radar_path)

    if len(mask) != len(radar_pc):
        raise ValueError(
            f"Mask and radar point count mismatch for {mask_name}: "
            f"mask={len(mask)}, radar={len(radar_pc)}"
        )

    return mask, radar_pc


def separate_static_and_moving(
    mask: np.ndarray,
    radar_pc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    moving_indices = mask != -1
    static_indices = mask == -1

    moving_mask = mask[moving_indices]
    static_mask = mask[static_indices]
    moving_radar_pc = radar_pc[moving_indices]
    static_radar_pc = radar_pc[static_indices]

    return moving_mask, static_mask, moving_radar_pc, static_radar_pc, moving_indices


def read_ego_pos(ego_pos_path: str | Path) -> np.ndarray:
    ego_pos_path = Path(ego_pos_path)

    if not ego_pos_path.exists():
        return np.zeros((0, 3), dtype=float)

    positions = []

    with ego_pos_path.open("r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            tx, ty, tz = map(float, parts[1:4])
            positions.append([tx, ty, tz])

    return np.asarray(positions, dtype=float)


def ego_pos_distance(ego_pos_path: str | Path) -> list[float]:
    ego_global_pos = read_ego_pos(ego_pos_path)

    if len(ego_global_pos) <= 1:
        return []

    distances = []

    for i in range(1, len(ego_global_pos)):
        d = np.linalg.norm(ego_global_pos[i, :2] - ego_global_pos[i - 1, :2])
        distances.append(float(d))

    return distances


def save_tracking_mask_to_csv(
    csv_path: str | Path,
    frame_id: int,
    mask: np.ndarray,
    cluster_track_dict: dict[int, int],
    moving_indices: np.ndarray,
    radar_points: np.ndarray,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    radar_points_x = radar_points[:, 0]
    radar_points_y = radar_points[:, 1]
    radar_points_z = radar_points[:, 2]

    radar_points_x_str = " ".join(map(str, radar_points_x))
    radar_points_y_str = " ".join(map(str, radar_points_y))
    radar_points_z_str = " ".join(map(str, radar_points_z))

    tracking_mask = np.full(mask.shape, -1, dtype=np.int32)
    tracking_mask[moving_indices] = -2

    for point_idx, cluster_id in enumerate(mask):
        cluster_id = int(cluster_id)
        if cluster_id in cluster_track_dict:
            tracking_mask[point_idx] = int(cluster_track_dict[cluster_id])

    mask_str = " ".join(map(str, tracking_mask))

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["id", "mask"])

        writer.writerow([frame_id, mask_str])
        writer.writerow([f"{frame_id}x", radar_points_x_str])
        writer.writerow([f"{frame_id}y", radar_points_y_str])
        writer.writerow([f"{frame_id}z", radar_points_z_str])


def parse_tracking_csv(csv_path: str | Path) -> dict[int, np.ndarray]:
    csv_path = Path(csv_path)

    rows: dict[int, np.ndarray] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames or "mask" not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns: id, mask. File: {csv_path}")

        for row in reader:
            row_id = str(row["id"])

            if row_id.endswith("x") or row_id.endswith("y") or row_id.endswith("z"):
                continue

            frame_id = int(row_id)
            if frame_id in rows:
                raise ValueError(f"Duplicate frame row found in CSV: {frame_id}")
            values = [int(float(v)) for v in str(row["mask"]).strip().split()]
            rows[frame_id] = np.asarray(values, dtype=np.int32)

    return rows
