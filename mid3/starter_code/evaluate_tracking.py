from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def is_frame_row(row_id: str) -> bool:
    row_id = str(row_id)
    return not (row_id.endswith("x") or row_id.endswith("y") or row_id.endswith("z"))


def parse_labels(text: str) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray([], dtype=np.int32)

    return np.asarray([int(float(v)) for v in str(text).strip().split()], dtype=np.int32)


def load_tracking_csv(csv_path: str | Path) -> dict[int, np.ndarray]:
    csv_path = Path(csv_path)

    rows: dict[int, np.ndarray] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames or "mask" not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns: id, mask. File: {csv_path}")

        for row in reader:
            row_id = str(row["id"])

            if not is_frame_row(row_id):
                continue

            frame_id = int(row_id)
            if frame_id in rows:
                raise ValueError(f"Duplicate frame row found in CSV: {frame_id}")
            rows[frame_id] = parse_labels(row["mask"])

    return rows


def evaluate(gt_csv: str | Path, pred_csv: str | Path) -> dict:
    gt = load_tracking_csv(gt_csv)
    pred = load_tracking_csv(pred_csv)

    frame_ids = sorted(gt.keys())

    fp_points = 0
    fn_points = 0
    idsw_proxy = 0
    fragmentation_proxy = 0
    gt_positive_points = 0

    prev_pred_for_gt: dict[int, int] = {}
    prev_gt_visible: dict[int, bool] = {}

    for frame_id in frame_ids:
        if frame_id not in pred:
            raise ValueError(f"Missing frame {frame_id} in prediction CSV.")

        gt_mask = gt[frame_id]
        pred_mask = pred[frame_id]

        if len(gt_mask) != len(pred_mask):
            raise ValueError(
                f"Length mismatch at frame {frame_id}: "
                f"gt={len(gt_mask)}, pred={len(pred_mask)}"
            )

        gt_pos = gt_mask >= 0
        pred_pos = pred_mask >= 0

        gt_positive_points += int(gt_pos.sum())

        fp_points += int(((~gt_pos) & pred_pos).sum())
        fn_points += int((gt_pos & (~pred_pos)).sum())
        # # ── debug：印前 20 幀的逐幀 FP/FN ──────────────────
        # if frame_id < 20:
        #     fp_this = int(((~gt_pos) & pred_pos).sum())
        #     fn_this = int((gt_pos & (~pred_pos)).sum())
        #     # FP 點對應的 pred track_id 有哪些
        #     fp_pred_ids = np.unique(pred_mask[(~gt_pos) & pred_pos]).tolist()
        #     print(f"frame {frame_id:3d} | "
        #           f"gt_pos={gt_pos.sum():5d}  pred_pos={pred_pos.sum():5d} | "
        #           f"FP={fp_this:5d}  FN={fn_this:4d} | "
        #           f"FP track_ids={fp_pred_ids}")
        # # ────────────────────────────────────────────────────

        current_gt_ids = set(int(v) for v in np.unique(gt_mask[gt_pos]))

        for gt_id in current_gt_ids:
            idx = gt_mask == gt_id
            pred_ids = pred_mask[idx]
            pred_ids = pred_ids[pred_ids >= 0]

            if len(pred_ids) == 0:
                prev_gt_visible[gt_id] = False
                continue

            values, counts = np.unique(pred_ids, return_counts=True)
            current_pred_id = int(values[np.argmax(counts)])

            if gt_id in prev_pred_for_gt and prev_pred_for_gt[gt_id] != current_pred_id:
                idsw_proxy += 1
                # print(f"  IDSW frame={frame_id} gt_id={gt_id} "
                #     f"prev_pred={prev_pred_for_gt[gt_id]} → curr_pred={current_pred_id}")
            if gt_id in prev_gt_visible and prev_gt_visible[gt_id] is False:
                fragmentation_proxy += 1

            prev_pred_for_gt[gt_id] = current_pred_id
            prev_gt_visible[gt_id] = True

        for gt_id in list(prev_gt_visible.keys()):
            if gt_id not in current_gt_ids:
                prev_gt_visible[gt_id] = False

    mota_proxy = 1.0 - (fp_points + fn_points + idsw_proxy) / max(gt_positive_points, 1)

    return {
        "frames": len(frame_ids),
        "gt_positive_points": gt_positive_points,
        "FP_points": fp_points,
        "FN_points": fn_points,
        "IDSW_proxy": idsw_proxy,
        "fragmentation_proxy": fragmentation_proxy,
        "MOTA_proxy": float(mota_proxy),
        "MOTA_style_score": float(mota_proxy),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = evaluate(args.gt, args.pred)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result]).to_csv(out_path, index=False)

    print(pd.DataFrame([result]).to_string(index=False))
    print(f"[DONE] saved: {out_path}")
