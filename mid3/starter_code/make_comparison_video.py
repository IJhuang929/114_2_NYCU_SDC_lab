"""
make_comparison_video.py — side-by-side ablation comparison video

Usage:
    python make_comparison_video.py \
        --left  ../outputs/ablation/seq3_full/seq_3_student_XXXXXXXX_XXXXXX/tracking_vis \
        --right ../outputs/ablation/seq3_no_ego/seq_3_student_XXXXXXXX_XXXXXX/tracking_vis \
        --out   ../outputs/ablation/comparison.mp4 \
        --fps   10

Options:
    --left      Path to tracking_vis folder (with ego compensation)
    --right     Path to tracking_vis folder (without ego compensation)
    --out       Output .mp4 path (default: comparison.mp4)
    --fps       Frames per second (default: 10)
    --label-left   Label for left panel (default: "With ego compensation")
    --label-right  Label for right panel (default: "Without ego compensation")
    --start     Start frame index (default: 0)
    --end       End frame index, -1 = all (default: -1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_frames(folder: Path) -> list[Path]:
    frames = sorted(folder.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frame_*.png found in {folder}")
    return frames


def add_label_bar(img: np.ndarray, label: str, color_bgr: tuple) -> np.ndarray:
    """Add a colored label bar at the top of an image."""
    bar_h = 36
    bar = np.full((bar_h, img.shape[1], 3), color_bgr, dtype=np.uint8)

    # White text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = (bar.shape[1] - tw) // 2
    ty = (bar_h + th) // 2 - 2
    cv2.putText(bar, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return np.vstack([bar, img])


def add_frame_counter(img: np.ndarray, frame_idx: int) -> np.ndarray:
    """Add frame number at bottom-right."""
    label = f"frame {frame_idx:04d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    x = img.shape[1] - tw - 10
    y = img.shape[0] - 8

    # Shadow for readability
    cv2.putText(img, label, (x + 1, y + 1), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (x, y), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def make_divider(h: int, w: int = 4) -> np.ndarray:
    """Thin vertical divider strip."""
    div = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)
    return div


def crop_panel(img: np.ndarray, panel: str) -> np.ndarray:
    """Crop to left or right half; 'full' returns the original."""
    if panel == "right":
        return img[:, img.shape[1] // 2:, :]
    if panel == "left":
        return img[:, : img.shape[1] // 2, :]
    return img


def build_frame(
    left_path: Path,
    right_path: Path,
    label_left: str,
    label_right: str,
    frame_idx: int,
    target_h: int,
    target_w: int,
    panel: str = "full",
) -> np.ndarray:
    def read_and_resize(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Cannot read image: {p}")
        img = crop_panel(img, panel)
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    left_img = read_and_resize(left_path)
    right_img = read_and_resize(right_path)

    # Teal bar for "with ego", Red bar for "without ego"
    left_img = add_label_bar(left_img, label_left,  color_bgr=(100, 140, 30))
    right_img = add_label_bar(right_img, label_right, color_bgr=(40, 60, 180))

    divider = make_divider(left_img.shape[0])
    combined = np.hstack([left_img, divider, right_img])
    combined = add_frame_counter(combined, frame_idx)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left",  required=True, help="tracking_vis folder (with ego comp)")
    parser.add_argument("--right", required=True, help="tracking_vis folder (no ego comp)")
    parser.add_argument("--out",   default="comparison.mp4")
    parser.add_argument("--fps",   type=int, default=10)
    parser.add_argument("--label-left",  default="With ego compensation")
    parser.add_argument("--label-right", default="Without ego compensation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=-1)
    parser.add_argument("--panel", default="full", choices=["full", "left", "right"],
                        help="Which panel to use from each frame image (default: full)")
    args = parser.parse_args()

    left_dir  = Path(args.left)
    right_dir = Path(args.right)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    left_frames  = load_frames(left_dir)
    right_frames = load_frames(right_dir)

    n = min(len(left_frames), len(right_frames))
    print(f"Left frames:  {len(left_frames)}")
    print(f"Right frames: {len(right_frames)}")
    print(f"Using:        {n} frames")

    start = args.start
    end   = n if args.end == -1 else min(args.end, n)
    frames_to_use = list(range(start, end))
    print(f"Rendering frames {start}..{end-1}  ({len(frames_to_use)} total)")

    # Determine output resolution from first frame
    probe = cv2.imread(str(left_frames[0]))
    if probe is None:
        raise ValueError(f"Cannot read first frame: {left_frames[0]}")
    probe = crop_panel(probe, args.panel)

    orig_h, orig_w = probe.shape[:2]
    # Scale down if too large (keep aspect ratio, max width per panel = 800px)
    max_panel_w = 1400
    if orig_w > max_panel_w:
        scale = max_panel_w / orig_w
        target_w = max_panel_w
        target_h = int(orig_h * scale)
    else:
        target_w = orig_w
        target_h = orig_h

    label_bar_h = 36
    total_h = target_h + label_bar_h
    total_w = target_w * 2 + 4  # 4px divider

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (total_w, total_h))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    print(f"Output resolution: {total_w} x {total_h}  @ {args.fps} fps")
    print(f"Writing to: {out_path}")

    for i, fi in enumerate(frames_to_use):
        frame = build_frame(
            left_frames[fi],
            right_frames[fi],
            args.label_left,
            args.label_right,
            fi,
            target_h,
            target_w,
            panel=args.panel,
        )
        writer.write(frame)

        if i % 50 == 0:
            print(f"  {i}/{len(frames_to_use)} frames written...")

    writer.release()
    print(f"\n[DONE] Saved: {out_path}")
    print(f"       Duration: {len(frames_to_use) / args.fps:.1f}s at {args.fps} fps")


if __name__ == "__main__":
    main()