"""
main.py — unified tracking entry point

Usage:
    python main.py --data-root <DATA_ROOT> --seq <SEQ> --output-root <OUTPUT_ROOT> [OPTIONS]

Required:
    --data-root     Path to dataset root (must contain seq_1/ seq_2/ seq_3/ seq_4/)
    --seq           Sequence to run: seq_1 | seq_2 | seq_3 | seq_4

    seq_1 / seq_2   Uses sdc_tracker_track1.Tracker
                    Offline post-processing: tracklet stitching + swap correction
                    CSV written after all frames are processed

    seq_3 / seq_4   Uses sdc_tracker_track2.Tracker with ego-motion compensation
                    CSV written per-frame incrementally

    --output-root   Directory where output folders are created

Optional:
    --no-save-csv           Skip writing result.csv
    --plot-every N          Save a visualization image every N frames (0 = disabled)
    --max-age N             Max frames a track survives without a match
                            (default: 5 for seq_1/2, 8 for seq_3/4)
    --min-hits N            Min consecutive hits before a track is confirmed
                            (default: 3 for seq_1/2, 1 for seq_3/4)
    --max-distance FLOAT    Max association distance gate in metres
                            (default: 5.0 for seq_1/2, 4.0 for seq_3/4)

Examples:
    python main.py --data-root ../public_dev --seq seq_1 --output-root ../outputs
    python main.py --data-root ../test_input --seq seq_2 --output-root ../outputs \\
        --max-age 8 --min-hits 1 --max-distance 5.0 --plot-every 10
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sdc_tracking_utils import (
    ego_pos_distance,
    list_frame_names,
    load_radar_and_mask_data,
    save_tracking_mask_to_csv,
    separate_static_and_moving,
)

# ============================================================
# Offline post-processing parameters (seq_1 / seq_2 only)
# ============================================================

ENABLE_OFFLINE_STITCHING = True
ENABLE_SWAP_CORRECTION = True

STITCH_MAX_GAP = 50
STITCH_MAX_DIST = 50.0
STITCH_MAX_RR_DIFF = 20.0

STITCH_GAP_WEIGHT = 0.15
STITCH_RR_WEIGHT = 0.5

SWAP_MAX_STEP_DIST = 6.0
SWAP_MARGIN = 2.0
SWAP_MIN_GAIN = 3.0


# ============================================================
# Ego motion loader (seq_3 / seq_4 only)
# ============================================================

def load_ego_delta_xy(ego_pos_path: Path):
    data = np.loadtxt(ego_pos_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ego_xy = data[:, 1:3].astype(float)
    global_delta = ego_xy[1:] - ego_xy[:-1]

    local_delta = []
    prev_heading = np.array([1.0, 0.0], dtype=float)

    for d in global_delta:
        norm = np.linalg.norm(d)

        if norm > 1e-6:
            heading = d / norm
            prev_heading = heading
        else:
            heading = prev_heading

        forward_axis = heading
        lateral_axis = np.array([-heading[1], heading[0]], dtype=float)

        forward_delta = float(np.dot(d, forward_axis))
        lateral_delta = float(np.dot(d, lateral_axis))

        local_delta.append([forward_delta, lateral_delta])

    return np.asarray(local_delta, dtype=float)


# ============================================================
# Shared helpers
# ============================================================

def calculate_cluster_centroids(
    moving_mask: np.ndarray,
    moving_radar_pc: np.ndarray,
):
    cluster_centroids = {}
    cluster_velocities = {}
    cluster_points_dict = {}
    cluster_rcs = {}

    valid_cluster_ids = np.unique(moving_mask)
    valid_cluster_ids = valid_cluster_ids[valid_cluster_ids >= 0]

    for cluster_id in valid_cluster_ids:
        cluster_id = int(cluster_id)

        points = moving_radar_pc[moving_mask == cluster_id]

        if len(points) == 0:
            continue

        centroid = np.mean(points[:, :3], axis=0)

        if points.shape[1] >= 4:
            mean_range_rate = float(np.mean(points[:, 3]))
        else:
            mean_range_rate = 0.0
        if points.shape[1] >= 5:
            mean_rcs = float(np.mean(points[:, 4]))
        else:
            mean_rcs = 0.0

        cluster_centroids[cluster_id] = centroid
        cluster_velocities[cluster_id] = mean_range_rate
        cluster_rcs[cluster_id] = mean_rcs
        cluster_points_dict[cluster_id] = points

    return cluster_centroids, cluster_velocities, cluster_points_dict, cluster_rcs


def generate_cluster_track_dict(
    confirmed_tracks: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    cluster_points_dict: dict[int, np.ndarray],
):
    cluster_track_dict = {}

    for track in confirmed_tracks:
        cluster_id = int(track[3])
        track_id = int(track[4])
        cluster_track_dict[cluster_id] = track_id

    return cluster_track_dict


def build_detections(
    cluster_centroids: dict[int, np.ndarray],
    cluster_velocities: dict[int, float],
    cluster_rcs: dict[int, float],
):
    detections = []

    for cluster_id, centroid in cluster_centroids.items():
        x, y, z = centroid
        range_rate = cluster_velocities.get(cluster_id, 0.0)
        rcs = cluster_rcs.get(cluster_id, 0.0)

        detections.append([
            float(x),
            float(y),
            float(z),
            int(cluster_id),
            float(range_rate),
            float(rcs),
        ])

    if len(detections) == 0:
        return np.empty((0, 6), dtype=float)

    return np.asarray(detections, dtype=float)


def detect_id_switches(
    confirmed_tracks: np.ndarray,
    prev_track_positions: dict[int, tuple[float, float]],
    max_dist: float = 8.0,
) -> set[int]:
    """Return track IDs that newly appeared within max_dist of a track that just disappeared.

    A track_id is flagged SW1 only when it's brand-new this frame AND it's spatially
    close to a track_id that existed last frame but is now gone — indicating the same
    physical object was re-assigned a different ID by the tracker.
    """
    if not prev_track_positions:
        return set()
    current_ids = {int(t[4]) for t in confirmed_tracks}
    disappeared = set(prev_track_positions) - current_ids
    if not disappeared:
        return set()
    switched: set[int] = set()
    for track in confirmed_tracks:
        tid = int(track[4])
        if tid in prev_track_positions:
            continue  # this track existed last frame, not a switch
        x, y = float(track[0]), float(track[1])
        for dead_id in disappeared:
            px, py = prev_track_positions[dead_id]
            if (x - px) ** 2 + (y - py) ** 2 < max_dist ** 2:
                switched.add(tid)
                break
    return switched


def trim_trajectory_segments(segs: list[list], max_pts: int = 60) -> None:
    """Remove oldest points from the first segment until total point count <= max_pts."""
    total = sum(len(s) for s in segs)
    while total > max_pts and segs:
        if segs[0]:
            segs[0].pop(0)
            total -= 1
            if not segs[0]:
                segs.pop(0)
        else:
            segs.pop(0)


def plot_tracklet_summary(
    tracklet_info: dict,
    id_remap: dict,
    total_frames: int,
    output_folder: Path,
    seq: str = "",
) -> None:
    if not tracklet_info:
        return

    import matplotlib.patches as mpatches

    sorted_ids = sorted(tracklet_info, key=lambda tid: tracklet_info[tid]["first_frame"])
    stitched_targets = set(id_remap.values())
    stitched_sources = set(id_remap.keys())
    mt_thresh = 0.8 * total_frames
    ml_thresh = 0.2 * total_frames

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(sorted_ids) * 0.4 + 2), 10))
    fig.suptitle(f"{seq} — Tracklet Analysis  ({total_frames} frames total)", fontsize=13)

    # top: individual tracklet lengths (before stitching collapse)
    colors_bar = []
    for tid in sorted_ids:
        if tid in stitched_targets:
            colors_bar.append("#4C96D7")
        elif tid in stitched_sources:
            colors_bar.append("#A0C4FF")
        else:
            colors_bar.append("#FF9A3C")

    ax1.bar(range(len(sorted_ids)),
            [tracklet_info[tid]["length"] for tid in sorted_ids],
            color=colors_bar, edgecolor="none", width=0.8)
    ax1.axhline(mt_thresh, color="green", linestyle="--", linewidth=1.5)
    ax1.axhline(ml_thresh, color="red",   linestyle="--", linewidth=1.5)
    ax1.set_xticks(range(len(sorted_ids)))
    ax1.set_xticklabels([str(t) for t in sorted_ids], rotation=90, fontsize=7)
    ax1.set_ylabel("Frames tracked")
    ax1.set_title("Individual tracklet length (before stitching)")
    ax1.legend(handles=[
        mpatches.Patch(color="#4C96D7", label="stitching target (canonical ID)"),
        mpatches.Patch(color="#A0C4FF", label="stitched fragment (merged into another)"),
        mpatches.Patch(color="#FF9A3C", label="standalone track"),
        plt.Line2D([0], [0], color="green", linestyle="--", label=f"MT ≥80% ({mt_thresh:.0f} f)"),
        plt.Line2D([0], [0], color="red",   linestyle="--", label=f"ML ≤20% ({ml_thresh:.0f} f)"),
    ], loc="upper right", fontsize=8)

    # bottom: canonical (stitched) coverage
    canonical_lengths: dict[int, int] = {}
    for tid, info in tracklet_info.items():
        root = id_remap.get(tid, tid)
        canonical_lengths[root] = canonical_lengths.get(root, 0) + info["length"]

    canon_ids = sorted(canonical_lengths,
                       key=lambda c: tracklet_info[c]["first_frame"] if c in tracklet_info else 0)
    canon_lens = [canonical_lengths[c] for c in canon_ids]

    mt_count = sum(1 for l in canon_lens if l >= mt_thresh)
    ml_count = sum(1 for l in canon_lens if l <= ml_thresh)

    bar_colors2 = [
        "#2ECC71" if l >= mt_thresh else ("#E74C3C" if l <= ml_thresh else "#95A5A6")
        for l in canon_lens
    ]
    ax2.bar(range(len(canon_ids)), canon_lens, color=bar_colors2, edgecolor="none", width=0.8)
    ax2.axhline(mt_thresh, color="green", linestyle="--", linewidth=1.5)
    ax2.axhline(ml_thresh, color="red",   linestyle="--", linewidth=1.5)
    ax2.set_xticks(range(len(canon_ids)))
    ax2.set_xticklabels([str(c) for c in canon_ids], rotation=90, fontsize=8)
    ax2.set_ylabel("Frames tracked (cumulative)")
    ax2.set_title(
        f"Effective coverage after stitching — "
        f"MT: {mt_count}  ML: {ml_count}  Partial: {len(canon_ids) - mt_count - ml_count}"
    )
    ax2.legend(handles=[
        mpatches.Patch(color="#2ECC71", label=f"MT (≥80%): {mt_count} tracks"),
        mpatches.Patch(color="#E74C3C", label=f"ML (≤20%): {ml_count} tracks"),
        mpatches.Patch(color="#95A5A6", label=f"Partial: {len(canon_ids)-mt_count-ml_count} tracks"),
    ], loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = output_folder / "tracklet_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] tracklet summary → {out_path}")


def assemble_video(output_folder: Path, fps: int = 10) -> Path | None:
    try:
        import cv2
    except ImportError:
        print("[WARN] opencv-python not installed; skipping video assembly")
        return None

    vis_dir = output_folder / "tracking_vis"
    frames = sorted(vis_dir.glob("frame_*.png"))

    if not frames:
        print("[WARN] No tracking_vis frames found; skipping video assembly")
        return None

    probe = cv2.imread(str(frames[0]))
    if probe is None:
        return None

    h, w = probe.shape[:2]
    out_path = output_folder / "tracking.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[VIDEO] assembling {len(frames)} frames → {out_path}  ({len(frames) / fps:.1f}s @ {fps}fps)")

    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(frames)} frames written...")

    writer.release()
    print(f"[DONE] video: {out_path}")
    return out_path


def visualize_tracking(
    static_radar_pc: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    confirmed_tracks: np.ndarray,
    track_trajectories: dict[int, list[list[tuple[float, float]]]],
    frame_idx: int,
    output_folder: Path,
    image_path: Path | None = None,
    seq: str = "",
    switched_track_ids: set[int] | None = None,
    swapped_track_ids: set[int] | None = None,
):
    """
    Optional visualization function.

    This function is not required for Kaggle submission.
    Students may modify it to inspect tracking behavior.
    """
    import matplotlib.image as mpimg
    output_img_folder = output_folder / "tracking_vis"
    output_img_folder.mkdir(parents=True, exist_ok=True)

    switched_track_ids = switched_track_ids or set()
    swapped_track_ids = swapped_track_ids or set()

    has_image = image_path is not None and Path(image_path).exists()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) if has_image else plt.subplots(1, 1, figsize=(10, 8))
    ax_cam, ax_track = (axes[0], axes[1]) if has_image else (None, axes)

    fig.suptitle(f"{seq} Tracking | Frame {frame_idx}", fontsize=14)
    colors = plt.cm.tab20.colors

    # left panel: camera image
    if has_image:
        ax_cam.imshow(mpimg.imread(str(image_path)))
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        ax_cam.set_title("Camera", fontsize=11)

    # right panel: radar tracking
    if static_radar_pc is not None and len(static_radar_pc) > 0:
        ax_track.scatter(-static_radar_pc[:, 1], static_radar_pc[:, 0], s=2, c="gray", alpha=0.3)
    ax_track.set_xlim(-60, 60)
    ax_track.set_ylim(0, 60)
    ax_track.set_xlabel("Lateral (-Y)")
    ax_track.set_ylabel("Forward (X)")
    ax_track.set_title("Tracking (track IDs + trajectories)")
    ax_track.grid(True, alpha=0.25)
    ax_track.text(0.99, 0.99, "static", transform=ax_track.transAxes,
                  ha="right", va="top", fontsize=8, color="gray")

    cluster_to_track = {int(t[3]): int(t[4]) for t in confirmed_tracks}

    for cluster_id, centroid in cluster_centroids.items():
        track_id = cluster_to_track.get(cluster_id)
        color = colors[track_id % len(colors)] if track_id is not None else (0.5, 0.5, 0.5)
        ax_track.scatter(-centroid[1], centroid[0], s=120, color=color, zorder=3)
        label = str(track_id) if track_id is not None else f"c{cluster_id}"
        ax_track.text(-centroid[1], centroid[0], label, fontsize=14, ha="center", va="bottom",
                      color=color, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, ec="none"))

        if track_id is not None:
            if track_id in switched_track_ids:
                ax_track.scatter(-centroid[1], centroid[0], marker="*", s=450,
                                 color="orange", zorder=5, linewidths=0)
                ax_track.text(-centroid[1], centroid[0] + 4, "SW1", fontsize=8,
                              color="orange", ha="center", fontweight="bold")
            if track_id in swapped_track_ids:
                ax_track.scatter(-centroid[1], centroid[0], marker="*", s=450,
                                 color="red", zorder=5, linewidths=0)
                ax_track.text(-centroid[1], centroid[0] + 4, "SW2", fontsize=8,
                              color="red", ha="center", fontweight="bold")

    for track_id, segments in track_trajectories.items():
        if not segments:
            continue
        color = colors[track_id % len(colors)]
        for seg_idx, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            pts = np.asarray(seg)
            is_last = (seg_idx == len(segments) - 1)
            ax_track.plot(-pts[:, 1], pts[:, 0],
                          linestyle="-" if is_last else "--",
                          linewidth=2.0 if is_last else 1.0,
                          alpha=0.9 if is_last else 0.4,
                          color=color)
        for seg in segments[:-1]:
            if seg:
                ax_track.scatter(-seg[-1][1], seg[-1][0], marker="x", s=80,
                                 color=color, linewidths=2, zorder=4)

    # proximity alert: yellow dashed line between two active tracks that are close
    PROXIMITY_M = 8.0
    track_list = [(int(t[4]), float(t[0]), float(t[1])) for t in confirmed_tracks]
    for i in range(len(track_list)):
        for j in range(i + 1, len(track_list)):
            _, xa, ya = track_list[i]
            _, xb, yb = track_list[j]
            if (xa - xb) ** 2 + (ya - yb) ** 2 < PROXIMITY_M ** 2:
                ax_track.plot([-ya, -yb], [xa, xb], color="yellow",
                              linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
                mx, my = -(ya + yb) / 2, (xa + xb) / 2
                ax_track.text(mx, my, "!", fontsize=11, ha="center", va="center",
                              color="yellow", fontweight="bold", zorder=6)

    legend_handles = [
        plt.scatter([], [], marker="*", s=200, color="orange",
                    label="SW1 (orange ★): new track ID appeared near a just-lost track"),
        plt.scatter([], [], marker="*", s=200, color="red",
                    label="SW2 (red ★): offline swap correction applied"),
        plt.scatter([], [], marker="x", s=80, color="gray", linewidths=2,
                    label="✕: trajectory break — track lost & reappeared"),
    ]
    ax_track.legend(handles=legend_handles, loc="lower right", fontsize=8,
                    framealpha=0.85, edgecolor="gray")

    fig.tight_layout()
    fig.savefig(output_img_folder / f"frame_{frame_idx:04d}.png", dpi=160)
    plt.close(fig)


# ============================================================
# Offline post-processing helpers (seq_1 / seq_2)
# ============================================================

def build_frame_track_states(
    frame_id: int,
    confirmed_tracks: np.ndarray,
    detections: np.ndarray,
):
    states = {}

    if confirmed_tracks is None or len(confirmed_tracks) == 0:
        return states

    det_by_cluster = {}

    for det in detections:
        cluster_id = int(det[3])
        det_by_cluster[cluster_id] = det

    for row in confirmed_tracks:
        x = float(row[0])
        y = float(row[1])
        cluster_id = int(row[3])
        track_id = int(row[4])

        det = det_by_cluster.get(cluster_id, None)
        rr = float(det[4]) if det is not None else 0.0

        states[track_id] = {
            "frame": frame_id,
            "xy": np.array([x, y], dtype=float),
            "rr": rr,
            "cluster_id": cluster_id,
        }

    return states


def update_tracklet_info(tracklet_info, frame_idx, confirmed_tracks, detections):
    if confirmed_tracks is None or len(confirmed_tracks) == 0:
        return

    det_by_cluster = {}

    for det in detections:
        cluster_id = int(det[3])
        det_by_cluster[cluster_id] = det

    for row in confirmed_tracks:
        x = float(row[0])
        y = float(row[1])
        cluster_id = int(row[3])
        track_id = int(row[4])

        det = det_by_cluster.get(cluster_id, None)

        if det is not None:
            rr = float(det[4])
            rcs = float(det[5])
        else:
            rr = 0.0
            rcs = 0.0

        if track_id not in tracklet_info:
            tracklet_info[track_id] = {
                "track_id": track_id,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "first_xy": np.array([x, y], dtype=float),
                "last_xy": np.array([x, y], dtype=float),
                "first_rr": rr,
                "last_rr": rr,
                "first_rcs": rcs,
                "last_rcs": rcs,
                "length": 1,
            }
        else:
            info = tracklet_info[track_id]
            info["last_frame"] = frame_idx
            info["last_xy"] = np.array([x, y], dtype=float)
            info["last_rr"] = rr
            info["last_rcs"] = rcs
            info["length"] += 1


def canonical_id(track_id: int, id_remap: dict[int, int]) -> int:
    seen = set()
    cur = int(track_id)

    while cur in id_remap and cur not in seen:
        seen.add(cur)
        cur = int(id_remap[cur])

    return cur


def build_tracklet_stitching_map(tracklet_info: dict[int, dict]):
    if not ENABLE_OFFLINE_STITCHING:
        return {}

    track_ids = sorted(tracklet_info.keys())
    candidates = []

    for a_id in track_ids:
        a = tracklet_info[a_id]

        for b_id in track_ids:
            if a_id == b_id:
                continue

            b = tracklet_info[b_id]
            gap = int(b["first_frame"]) - int(a["last_frame"])

            if gap <= 0 or gap > STITCH_MAX_GAP:
                continue

            dist = float(np.linalg.norm(a["last_xy"] - b["first_xy"]))

            if dist > STITCH_MAX_DIST:
                continue

            rr_diff = abs(float(a["last_rr"]) - float(b["first_rr"]))

            if rr_diff > STITCH_MAX_RR_DIFF:
                continue

            score = dist + STITCH_RR_WEIGHT * rr_diff + STITCH_GAP_WEIGHT * gap

            candidates.append({
                "from_id": int(b_id),
                "to_id": int(a_id),
                "score": score,
                "gap": gap,
                "dist": dist,
                "rr_diff": rr_diff,
            })

    candidates.sort(key=lambda x: x["score"])

    id_remap = {}
    used_from = set()

    for cand in candidates:
        from_id = int(cand["from_id"])
        to_id = int(cand["to_id"])

        if from_id in used_from:
            continue

        root_from = canonical_id(from_id, id_remap)
        root_to = canonical_id(to_id, id_remap)

        if root_from == root_to:
            continue

        from_info = tracklet_info[from_id]
        to_info = tracklet_info[to_id]

        if int(from_info["first_frame"]) <= int(to_info["first_frame"]):
            continue

        id_remap[root_from] = root_to
        used_from.add(from_id)

    flat_remap = {}

    for tid in track_ids:
        root = canonical_id(tid, id_remap)

        if root != tid:
            flat_remap[int(tid)] = int(root)

    print("[INFO] offline stitching remap:")

    if len(flat_remap) == 0:
        print("  no remap")
    else:
        for from_id, to_id in sorted(flat_remap.items()):
            print(f"  {from_id} -> {to_id}")

    return flat_remap


def apply_id_remap(cluster_track_dict: dict[int, int], id_remap: dict[int, int]):
    if not id_remap:
        return cluster_track_dict

    fixed = {}

    for cluster_id, track_id in cluster_track_dict.items():
        fixed[cluster_id] = canonical_id(int(track_id), id_remap)

    return fixed


def apply_id_remap_to_frame_track_states(
    frame_track_states: dict[int, dict[int, dict]],
    id_remap: dict[int, int],
):
    if not id_remap:
        return frame_track_states

    new_frame_track_states = {}

    for frame_id, states in frame_track_states.items():
        new_states = {}

        for track_id, state in states.items():
            new_id = canonical_id(int(track_id), id_remap)
            new_state = dict(state)
            new_state["track_id"] = new_id

            if new_id not in new_states:
                new_states[new_id] = new_state

        new_frame_track_states[frame_id] = new_states

    return new_frame_track_states


def apply_swap_to_future_records(
    all_frame_records: list[dict],
    start_frame: int,
    id_a: int,
    id_b: int,
):
    for record in all_frame_records:
        if int(record["frame_id"]) < start_frame:
            continue

        d = record["cluster_track_dict"]

        for cluster_id, track_id in list(d.items()):
            if int(track_id) == int(id_a):
                d[cluster_id] = int(id_b)
            elif int(track_id) == int(id_b):
                d[cluster_id] = int(id_a)


def offline_swap_correction(
    all_frame_records: list[dict],
    frame_track_states: dict[int, dict[int, dict]],
):
    if not ENABLE_SWAP_CORRECTION:
        return 0, []

    swap_count = 0
    swap_events: list[tuple[int, int, int]] = []
    frame_ids = sorted(frame_track_states.keys())

    for idx in range(len(frame_ids) - 1):
        f0 = frame_ids[idx]
        f1 = frame_ids[idx + 1]

        if f1 != f0 + 1:
            continue

        prev_states = frame_track_states.get(f0, {})
        next_states = frame_track_states.get(f1, {})

        common_ids = sorted(set(prev_states.keys()) & set(next_states.keys()))

        if len(common_ids) < 2:
            continue

        best_swap = None
        best_gain = 0.0

        for i in range(len(common_ids)):
            for j in range(i + 1, len(common_ids)):
                id_a = common_ids[i]
                id_b = common_ids[j]

                a0 = prev_states[id_a]["xy"]
                b0 = prev_states[id_b]["xy"]
                a1 = next_states[id_a]["xy"]
                b1 = next_states[id_b]["xy"]

                keep_cost = np.linalg.norm(a0 - a1) + np.linalg.norm(b0 - b1)
                swap_cost = np.linalg.norm(a0 - b1) + np.linalg.norm(b0 - a1)

                if np.linalg.norm(a0 - b1) > SWAP_MAX_STEP_DIST:
                    continue

                if np.linalg.norm(b0 - a1) > SWAP_MAX_STEP_DIST:
                    continue

                gain = keep_cost - swap_cost

                if gain > best_gain and swap_cost + SWAP_MARGIN < keep_cost:
                    best_gain = gain
                    best_swap = (id_a, id_b, f1, keep_cost, swap_cost)

        if best_swap is None or best_gain < SWAP_MIN_GAIN:
            continue

        id_a, id_b, start_frame, keep_cost, swap_cost = best_swap

        print(
            f"[DEBUG] swap correction at frame {start_frame}: "
            f"{id_a} <-> {id_b}, "
            f"keep={keep_cost:.2f}, swap={swap_cost:.2f}, "
            f"gain={best_gain:.2f}"
        )

        swap_events.append((start_frame, id_a, id_b))

        apply_swap_to_future_records(
            all_frame_records=all_frame_records,
            start_frame=start_frame,
            id_a=id_a,
            id_b=id_b,
        )

        for future_f in frame_ids:
            if future_f < start_frame:
                continue

            states = frame_track_states.get(future_f, {})
            has_a = id_a in states
            has_b = id_b in states

            if has_a and has_b:
                states[id_a], states[id_b] = states[id_b], states[id_a]
            elif has_a:
                states[id_b] = states.pop(id_a)
            elif has_b:
                states[id_a] = states.pop(id_b)

        swap_count += 1

    return swap_count, swap_events


# ============================================================
# Sequence runners
# ============================================================

def run_sequence_track1(
    data_root: Path,
    seq: str,
    output_root: Path,
    save_csv: bool,
    plot_every: int,
    max_age: int,
    min_hits: int,
    max_distance: float,
    make_video: bool = False,
    video_fps: int = 10,
):
    """seq_1 / seq_2: offline stitching + swap correction, no ego compensation."""
    from sdc_tracker_track1 import Tracker

    seq_dir = data_root / seq
    mask_dir = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"

    if not mask_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError(f"Missing radar/ or mask_cluster/ under {seq_dir}")

    output_folder = output_root / f"{seq}_student_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)
    result_csv = output_folder / "result.csv"

    if result_csv.exists():
        result_csv.unlink()

    tracker = Tracker(
        max_age=max_age,
        min_hits=min_hits,
        max_distance=max_distance,
    )

    track_trajectories: dict[int, list[list[tuple[float, float]]]] = {}
    prev_active_ids: set[int] = set()
    prev_track_positions: dict[int, tuple[float, float]] = {}
    vis_data_by_frame: dict[int, dict] = {}
    frame_names = list_frame_names(mask_dir)
    all_frame_records = []
    tracklet_info = {}
    frame_track_states = {}

    print("============================================================")
    print(f"Run tracking (seq_1/seq_2 mode) — {seq}")
    print("============================================================")
    print(f"frames: {len(frame_names)}, output: {output_folder}")
    print(f"max_age: {max_age}, min_hits: {min_hits}, max_distance: {max_distance}")

    for frame_idx, mask_name in enumerate(frame_names):
        mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)

        moving_mask, static_mask, moving_radar_pc, static_radar_pc, moving_indices = (
            separate_static_and_moving(mask, radar_pc)
        )

        cluster_centroids, cluster_velocities, cluster_points_dict, cluster_rcs = (
            calculate_cluster_centroids(moving_mask, moving_radar_pc)
        )

        detections = build_detections(cluster_centroids, cluster_velocities, cluster_rcs)
        confirmed_tracks = tracker.update(detections)

        cluster_track_dict = generate_cluster_track_dict(
            confirmed_tracks, cluster_centroids, cluster_points_dict
        )

        frame_track_states[frame_idx] = build_frame_track_states(
            frame_id=frame_idx,
            confirmed_tracks=confirmed_tracks,
            detections=detections,
        )

        switched_track_ids = detect_id_switches(
            confirmed_tracks, prev_track_positions, max_dist=max_distance
        )
        prev_track_positions = {int(t[4]): (float(t[0]), float(t[1])) for t in confirmed_tracks}

        active_track_ids = set()

        for track in confirmed_tracks:
            track_id = int(track[4])
            x, y = float(track[0]), float(track[1])
            active_track_ids.add(track_id)

            if track_id not in track_trajectories:
                track_trajectories[track_id] = [[]]
            elif track_id not in prev_active_ids:
                track_trajectories[track_id].append([])

            track_trajectories[track_id][-1].append((x, y))
            trim_trajectory_segments(track_trajectories[track_id])

        for tid in list(track_trajectories.keys()):
            if tid not in active_track_ids:
                segs = track_trajectories[tid]
                if segs and segs[0]:
                    segs[0].pop(0)
                    if not segs[0]:
                        segs.pop(0)
                if not segs:
                    del track_trajectories[tid]

        prev_active_ids = set(active_track_ids)

        if save_csv:
            all_frame_records.append({
                "frame_id": frame_idx,
                "mask": mask.copy(),
                "cluster_track_dict": dict(cluster_track_dict),
                "moving_indices": moving_indices.copy(),
                "radar_points": radar_pc.copy(),
            })

            update_tracklet_info(
                tracklet_info=tracklet_info,
                frame_idx=frame_idx,
                confirmed_tracks=confirmed_tracks,
                detections=detections,
            )

        if plot_every > 0 and frame_idx % plot_every == 0:
            image_path = seq_dir / "image" / (Path(mask_name).stem + ".jpg")
            vis_data_by_frame[frame_idx] = {
                "static_radar_pc": static_radar_pc,
                "cluster_centroids": dict(cluster_centroids),
                "confirmed_tracks": confirmed_tracks,
                "track_trajectories": copy.deepcopy(track_trajectories),
                "switched_track_ids": set(switched_track_ids),
                "image_path": image_path,
            }
            visualize_tracking(
                static_radar_pc=static_radar_pc,
                cluster_centroids=cluster_centroids,
                confirmed_tracks=confirmed_tracks,
                track_trajectories=track_trajectories,
                frame_idx=frame_idx,
                output_folder=output_folder,
                image_path=image_path,
                seq=seq,
                switched_track_ids=switched_track_ids,
            )

        if frame_idx % 50 == 0:
            print(
                f"  frame {frame_idx:04d}: "
                f"detections={len(detections)}, confirmed={len(confirmed_tracks)}"
            )

    if save_csv:
        id_remap = build_tracklet_stitching_map(tracklet_info)

        print(f"[INFO] tracklets: {len(tracklet_info)}")
        print(f"[INFO] stitched ids: {len(id_remap)}")

        for record in all_frame_records:
            record["cluster_track_dict"] = apply_id_remap(
                record["cluster_track_dict"], id_remap
            )

        frame_track_states = apply_id_remap_to_frame_track_states(
            frame_track_states, id_remap
        )

        swap_count, swap_events = offline_swap_correction(
            all_frame_records=all_frame_records,
            frame_track_states=frame_track_states,
        )
        print(f"[INFO] swap corrections: {swap_count}")

        plot_tracklet_summary(
            tracklet_info=tracklet_info,
            id_remap=id_remap,
            total_frames=len(frame_names),
            output_folder=output_folder,
            seq=seq,
        )

        if plot_every > 0 and swap_events and vis_data_by_frame:
            plotted = sorted(vis_data_by_frame.keys())
            swapped_by_frame: dict[int, set[int]] = {}
            for ev_frame, id_a, id_b in swap_events:
                nearest = min(plotted, key=lambda f: abs(f - ev_frame))
                if abs(nearest - ev_frame) < max(plot_every, 1):
                    swapped_by_frame.setdefault(nearest, set()).update([id_a, id_b])
            for frame_id, swapped_ids in swapped_by_frame.items():
                vd = vis_data_by_frame[frame_id]
                visualize_tracking(
                    static_radar_pc=vd["static_radar_pc"],
                    cluster_centroids=vd["cluster_centroids"],
                    confirmed_tracks=vd["confirmed_tracks"],
                    track_trajectories=vd["track_trajectories"],
                    frame_idx=frame_id,
                    output_folder=output_folder,
                    image_path=vd["image_path"],
                    seq=seq,
                    switched_track_ids=vd["switched_track_ids"],
                    swapped_track_ids=swapped_ids,
                )

        if result_csv.exists():
            result_csv.unlink()

        for record in all_frame_records:
            save_tracking_mask_to_csv(
                csv_path=result_csv,
                frame_id=record["frame_id"],
                mask=record["mask"],
                cluster_track_dict=record["cluster_track_dict"],
                moving_indices=record["moving_indices"],
                radar_points=record["radar_points"],
            )

    print(f"[DONE] output folder: {output_folder}")

    if save_csv:
        print(f"[DONE] result csv: {result_csv}")

    if make_video and plot_every > 0:
        assemble_video(output_folder, fps=video_fps)

    return output_folder


def run_sequence_ego(
    data_root: Path,
    seq: str,
    output_root: Path,
    save_csv: bool,
    plot_every: int,
    max_age: int,
    min_hits: int,
    max_distance: float,
    make_video: bool = False,
    video_fps: int = 10,
):
    """seq_3 / seq_4: ego compensation, immediate per-frame CSV write."""
    from sdc_tracker_track2 import Tracker

    seq_dir = data_root / seq
    mask_dir = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"

    if not mask_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError(f"Missing radar/ or mask_cluster/ under {seq_dir}")

    output_folder = output_root / f"{seq}_student_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)
    result_csv = output_folder / "result.csv"

    if result_csv.exists():
        result_csv.unlink()

    ego_pos_path = seq_dir / "ego_global_pos.txt"
    ego_motion = load_ego_delta_xy(ego_pos_path)
    print(f"ego_global_pos: {ego_pos_path}")
    print(f"ego motion steps loaded: {len(ego_motion)}")
    print("ego compensation is enabled.")

    if len(ego_motion) > 0:
        print("ego_delta[0] =", ego_motion[0])

    tracker = Tracker(
        max_age=max_age,
        min_hits=min_hits,
        max_distance=max_distance,
    )

    track_trajectories: dict[int, list[list[tuple[float, float]]]] = {}
    prev_active_ids: set[int] = set()
    prev_track_positions: dict[int, tuple[float, float]] = {}
    frame_names = list_frame_names(mask_dir)

    print("============================================================")
    print(f"Run tracking (seq_3/seq_4 mode) — {seq}")
    print("============================================================")
    print(f"frames: {len(frame_names)}, output: {output_folder}")
    print(f"max_age: {max_age}, min_hits: {min_hits}, max_distance: {max_distance}")

    for frame_idx, mask_name in enumerate(frame_names):
        mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)

        moving_mask, static_mask, moving_radar_pc, static_radar_pc, moving_indices = (
            separate_static_and_moving(mask, radar_pc)
        )

        cluster_centroids, cluster_velocities, cluster_points_dict, cluster_rcs = (
            calculate_cluster_centroids(moving_mask, moving_radar_pc)
        )

        detections = build_detections(cluster_centroids, cluster_velocities, cluster_rcs)

        ego_delta = None

        if frame_idx > 0 and frame_idx - 1 < len(ego_motion):
            ego_delta = ego_motion[frame_idx - 1]

            if frame_idx == 1:
                print("ego_motion[0] =", ego_delta)

        confirmed_tracks = tracker.update(detections, ego_delta=ego_delta)

        cluster_track_dict = {int(t[3]): int(t[4]) for t in confirmed_tracks}

        switched_track_ids = detect_id_switches(
            confirmed_tracks, prev_track_positions, max_dist=max_distance
        )
        prev_track_positions = {int(t[4]): (float(t[0]), float(t[1])) for t in confirmed_tracks}

        active_track_ids = set()

        for track in confirmed_tracks:
            track_id = int(track[4])
            x, y = float(track[0]), float(track[1])
            active_track_ids.add(track_id)

            if track_id not in track_trajectories:
                track_trajectories[track_id] = [[]]
            elif track_id not in prev_active_ids:
                track_trajectories[track_id].append([])

            track_trajectories[track_id][-1].append((x, y))
            trim_trajectory_segments(track_trajectories[track_id])

        for tid in list(track_trajectories.keys()):
            if tid not in active_track_ids:
                segs = track_trajectories[tid]
                if segs and segs[0]:
                    segs[0].pop(0)
                    if not segs[0]:
                        segs.pop(0)
                if not segs:
                    del track_trajectories[tid]

        prev_active_ids = set(active_track_ids)

        if save_csv:
            save_tracking_mask_to_csv(
                csv_path=result_csv,
                frame_id=frame_idx,
                mask=mask,
                cluster_track_dict=cluster_track_dict,
                moving_indices=moving_indices,
                radar_points=radar_pc,
            )

        if plot_every > 0 and frame_idx % plot_every == 0:
            image_path = seq_dir / "image" / (Path(mask_name).stem + ".jpg")
            visualize_tracking(
                static_radar_pc=static_radar_pc,
                cluster_centroids=cluster_centroids,
                confirmed_tracks=confirmed_tracks,
                track_trajectories=track_trajectories,
                frame_idx=frame_idx,
                output_folder=output_folder,
                image_path=image_path,
                seq=seq,
                switched_track_ids=switched_track_ids,
            )

        if frame_idx % 50 == 0:
            print(
                f"  frame {frame_idx:04d}: "
                f"detections={len(detections)}, confirmed={len(confirmed_tracks)}"
            )

    print(f"[DONE] output folder: {output_folder}")

    if save_csv:
        print(f"[DONE] result csv: {result_csv}")

    if make_video and plot_every > 0:
        assemble_video(output_folder, fps=video_fps)

    return output_folder


# ============================================================
# Entry point
# ============================================================

SEQ1_DEFAULTS = dict(max_age=8, min_hits=1, max_distance=5.0)
SEQ3_DEFAULTS = dict(max_age=8, min_hits=1, max_distance=5.0)


def run_sequence(
    data_root: str,
    seq: str,
    output_root: str,
    save_csv: bool,
    plot_every: int,
    max_age: int | None,
    min_hits: int | None,
    max_distance: float | None,
    make_video: bool = False,
    video_fps: int = 10,
):
    data_root = Path(data_root)
    output_root = Path(output_root)

    if seq in ["seq_1", "seq_2"]:
        defaults = SEQ1_DEFAULTS
        runner = run_sequence_track1
    else:
        defaults = SEQ3_DEFAULTS
        runner = run_sequence_ego

    return runner(
        data_root=data_root,
        seq=seq,
        output_root=output_root,
        save_csv=save_csv,
        plot_every=plot_every,
        max_age=max_age if max_age is not None else defaults["max_age"],
        min_hits=min_hits if min_hits is not None else defaults["min_hits"],
        max_distance=max_distance if max_distance is not None else defaults["max_distance"],
        make_video=make_video,
        video_fps=video_fps,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--output-root", required=True)

    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Disable writing result.csv.",
    )
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--make-video", action="store_true",
                        help="Assemble tracking_vis frames into tracking.mp4 after plotting.")
    parser.add_argument("--video-fps", type=int, default=10,
                        help="FPS for --make-video output (default: 10).")

    parser.add_argument("--max-age", type=int, default=None)
    parser.add_argument("--min-hits", type=int, default=None)
    parser.add_argument("--max-distance", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_sequence(
        data_root=args.data_root,
        seq=args.seq,
        output_root=args.output_root,
        save_csv=not args.no_save_csv,
        plot_every=args.plot_every,
        max_age=args.max_age,
        min_hits=args.min_hits,
        max_distance=args.max_distance,
        make_video=args.make_video,
        video_fps=args.video_fps,
    )
