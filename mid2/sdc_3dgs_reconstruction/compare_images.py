"""
Side-by-side comparison of test_submission vs original images.
Matches each submission PNG to the closest-timestamp original JPG.

Usage:
  python compare_images.py [--track 1] [--auto] [--fps 4]

Controls:
  Left/Right arrow  : prev / next pair
  Space             : toggle auto-play
  Q / Esc           : quit
"""

import argparse
import bisect
import os
import sys

import cv2
import numpy as np


def load_timestamps(directory, ext):
    files = [f for f in os.listdir(directory) if f.endswith(ext)]
    # filename (without ext) is the nanosecond timestamp
    pairs = sorted((int(os.path.splitext(f)[0]), os.path.join(directory, f)) for f in files)
    return pairs  # list of (timestamp_ns, path)


def find_closest(ts, sorted_pairs):
    """Return path of the entry in sorted_pairs closest to ts."""
    timestamps = [p[0] for p in sorted_pairs]
    idx = bisect.bisect_left(timestamps, ts)
    candidates = []
    if idx < len(sorted_pairs):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates, key=lambda i: abs(sorted_pairs[i][0] - ts))
    return sorted_pairs[best][1], abs(sorted_pairs[best][0] - ts)


def build_pairs(submission_dir, image_dir):
    sub_pairs = load_timestamps(submission_dir, ".png")
    img_pairs = load_timestamps(image_dir, ".jpg")
    if not sub_pairs:
        sys.exit(f"No .png files found in {submission_dir}")
    if not img_pairs:
        sys.exit(f"No .jpg files found in {image_dir}")

    result = []
    for ts, sub_path in sub_pairs:
        orig_path, diff_ns = find_closest(ts, img_pairs)
        result.append((ts, sub_path, orig_path, diff_ns))
    return result


def make_side_by_side(sub_path, orig_path, ts_ns, diff_ns, target_h=720):
    sub = cv2.imread(sub_path)
    orig = cv2.imread(orig_path)
    if sub is None or orig is None:
        return None

    # Resize both to same height
    def resize_h(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    sub_r = resize_h(sub, target_h)
    orig_r = resize_h(orig, target_h)

    # Add a 4px white divider
    divider = np.full((target_h, 4, 3), 255, dtype=np.uint8)
    canvas = np.hstack([orig_r, divider, sub_r])

    # Labels
    label_h = 36
    header = np.zeros((label_h, canvas.shape[1], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, "Original", (10, 26), font, 0.9, (200, 230, 255), 2)
    right_x = orig_r.shape[1] + 4 + 10
    cv2.putText(header, "Submission", (right_x, 26), font, 0.9, (200, 255, 200), 2)

    # Footer: timestamp info
    footer = np.zeros((label_h, canvas.shape[1], 3), dtype=np.uint8)
    info = f"ts={ts_ns}  |  dt={diff_ns/1e6:.1f}ms"
    cv2.putText(footer, info, (10, 26), font, 0.7, (180, 180, 180), 1)

    return np.vstack([header, canvas, footer])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--auto", action="store_true", help="Start in auto-play mode")
    parser.add_argument("--fps", type=float, default=4.0, help="Auto-play speed")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "mid2dataset", f"Track{args.track}")
    submission_dir = os.path.join(base, "test_submission")
    image_dir = os.path.join(base, "data", "image")

    print(f"Track {args.track}")
    print(f"  submission : {submission_dir}")
    print(f"  originals  : {image_dir}")

    pairs = build_pairs(submission_dir, image_dir)
    print(f"  matched {len(pairs)} pairs")

    idx = 0
    auto_play = args.auto
    delay_ms = max(1, int(1000 / args.fps))
    window = f"Track{args.track} | orig (left) vs submission (right)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ts, sub_path, orig_path, diff_ns = pairs[idx]
        frame = make_side_by_side(sub_path, orig_path, ts, diff_ns)
        if frame is None:
            print(f"  [warn] could not load images at index {idx}")
        else:
            title = f"[{idx+1}/{len(pairs)}] Track{args.track} | orig vs submission"
            cv2.setWindowTitle(window, title)
            cv2.imshow(window, frame)

        wait = delay_ms if auto_play else 0
        key = cv2.waitKey(wait) & 0xFF

        if key in (ord('q'), 27):  # Q or Esc
            break
        elif key == ord(' '):
            auto_play = not auto_play
        elif key == 81 or key == ord('a'):  # left arrow or a
            idx = max(0, idx - 1)
        elif key == 83 or key == ord('d'):  # right arrow or d
            idx = min(len(pairs) - 1, idx + 1)
        elif auto_play:
            idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
