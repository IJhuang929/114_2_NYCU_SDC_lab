from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sdc_tracking_utils import list_frame_names, load_radar_and_mask_data


def parse_values(text: str, dtype=float):
    if text is None or str(text).strip() == "":
        return []

    values = []
    for v in str(text).strip().split():
        if dtype is int:
            values.append(int(float(v)))
        else:
            values.append(dtype(v))
    return values


def load_submission_rows(submission_csv: str | Path) -> dict[str, str]:
    submission_csv = Path(submission_csv)

    rows = {}

    with submission_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames or "mask" not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns: id, mask. File: {submission_csv}")

        for row in reader:
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"Duplicate row id found in submission: {row_id}")
            rows[row_id] = str(row["mask"])

    return rows


def check_submission(submission_csv: str | Path, data_root: str | Path, seq: str) -> None:
    data_root = Path(data_root)
    seq_dir = data_root / seq

    mask_dir = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"

    if not mask_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError(f"Missing radar/ or mask_cluster/ under {seq_dir}")

    rows = load_submission_rows(submission_csv)
    frame_names = list_frame_names(mask_dir)

    errors = []

    for frame_id, mask_name in enumerate(frame_names):
        expected_ids = [
            str(frame_id),
            f"{frame_id}x",
            f"{frame_id}y",
            f"{frame_id}z",
        ]

        for row_id in expected_ids:
            if row_id not in rows:
                errors.append(f"Missing row id: {row_id}")

        if any(row_id not in rows for row_id in expected_ids):
            continue

        _, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)
        n_points = len(radar_pc)

        pred_mask = parse_values(rows[str(frame_id)], dtype=int)
        xs = parse_values(rows[f"{frame_id}x"], dtype=float)
        ys = parse_values(rows[f"{frame_id}y"], dtype=float)
        zs = parse_values(rows[f"{frame_id}z"], dtype=float)

        if len(pred_mask) != n_points:
            errors.append(
                f"Frame {frame_id}: mask length mismatch, "
                f"expected {n_points}, got {len(pred_mask)}"
            )

        if len(xs) != n_points:
            errors.append(
                f"Frame {frame_id}: x length mismatch, "
                f"expected {n_points}, got {len(xs)}"
            )

        if len(ys) != n_points:
            errors.append(
                f"Frame {frame_id}: y length mismatch, "
                f"expected {n_points}, got {len(ys)}"
            )

        if len(zs) != n_points:
            errors.append(
                f"Frame {frame_id}: z length mismatch, "
                f"expected {n_points}, got {len(zs)}"
            )

        if len(pred_mask) == n_points:
            invalid = [v for v in pred_mask if v < -2]
            if invalid:
                errors.append(f"Frame {frame_id}: found labels smaller than -2")

    expected_row_count = len(frame_names) * 4
    actual_row_count = len(rows)

    if actual_row_count != expected_row_count:
        errors.append(
            f"Total row count mismatch: expected {expected_row_count}, got {actual_row_count}"
        )

    if errors:
        print("[FAILED] submission format check failed")
        for msg in errors[:50]:
            print(" -", msg)

        if len(errors) > 50:
            print(f" - ... {len(errors) - 50} more errors")

        raise SystemExit(1)

    print("[OK] submission format is valid")
    print(f"submission: {submission_csv}")
    print(f"sequence: {seq}")
    print(f"frames: {len(frame_names)}")
    print(f"rows: {actual_row_count}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_submission(
        submission_csv=args.submission,
        data_root=args.data_root,
        seq=args.seq,
    )
