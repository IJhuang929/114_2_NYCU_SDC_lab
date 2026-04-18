#!/usr/bin/env python3
"""
Grid Search for radar_preprocessor.py parameters — TensorBoard edition.

Real-time TensorBoard logging: monitor progress live while search runs.
  tensorboard --logdir=runs/grid_search --port=6006 --bind_all

Strategy:
  Phase 1 — Random search: sample N combos from full 3^8=6561 grid
  Phase 2 — Fine search:   for top-K combos, search +-1 step neighbours

Usage (inside the Docker / catkin_ws environment):
  python3 scripts/grid_search.py [--n-random 100] [--n-eval-frames 50]
"""

import os
import sys
import csv
import glob
import time
import struct
import random
import signal
import shutil
import binascii
import argparse
import subprocess
import itertools
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
PARAMS_YAML = os.path.join(PKG_DIR, "config", "params.yaml")
PREPROCESSOR = os.path.join(PKG_DIR, "processing_codes", "radar_preprocessor.py")
RESULT_DIR = os.path.join(PKG_DIR, "result")
RUNS_DIR = os.path.join(PKG_DIR, "runs", "grid_search")

# ---------------------------------------------------------------------------
#  Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "radar_z_min":          [-6.0, -4.0],
    "radar_z_max":          [15.0, 30.0],
    "range_min":            [2.0, 3.0],
    "range_max":            [50.0, 70.0],
    "accumulate_frames":    [3, 5],
    "accumulate_voxel_size": [0.3, 0.5],
    "radar_sol_mean_k":     [10, 20],
    "radar_sol_std_dev_mul": [1.0, 2.0],
}
PARAM_NAMES = list(SEARCH_SPACE.keys())
METRIC_NAMES = ["mean_fitness", "radar_pts_mean", "failed_ratio", "composite_score"]

# ============================================================================
#  TensorBoard writer (no torch dependency)
# ============================================================================
class TBWriter:
    """Lightweight TensorBoard event-file writer using tensorboard protobuf."""

    def __init__(self, logdir):
        from tensorboard.compat.proto import event_pb2, summary_pb2
        self._event_pb2 = event_pb2
        self._summary_pb2 = summary_pb2
        os.makedirs(logdir, exist_ok=True)
        fname = os.path.join(
            logdir,
            f"events.out.tfevents.{int(time.time())}.grid_search",
        )
        self._f = open(fname, "wb")
        self._write_event(event_pb2.Event(file_version="brain.Event:2"))

    # --- low-level TFRecord encoding ---
    def _write_event(self, event):
        data = event.SerializeToString()
        hdr = struct.pack("Q", len(data))
        self._f.write(hdr + struct.pack("I", self._crc(hdr))
                       + data + struct.pack("I", self._crc(data)))
        self._f.flush()

    @staticmethod
    def _crc(data):
        crc = binascii.crc32(data) & 0xFFFFFFFF
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    # --- public API ---
    def add_scalar(self, tag, value, step):
        s = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(
                tag=tag, simple_value=float(value))])
        self._write_event(self._event_pb2.Event(
            wall_time=time.time(), step=step, summary=s))

    def add_image(self, tag, img_array, step):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img_array).save(buf, format="PNG")
        h, w = img_array.shape[:2]
        img_proto = self._summary_pb2.Summary.Image(
            height=h, width=w, colorspace=3,
            encoded_image_string=buf.getvalue())
        s = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(tag=tag, image=img_proto)])
        self._write_event(self._event_pb2.Event(
            wall_time=time.time(), step=step, summary=s))

    def add_text(self, tag, text, step):
        from tensorboard.compat.proto import (
            tensor_pb2, tensor_shape_pb2, types_pb2)
        tensor = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            string_val=[text.encode("utf-8")],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[]))
        plugin = self._summary_pb2.SummaryMetadata.PluginData(
            plugin_name="text")
        smd = self._summary_pb2.SummaryMetadata(plugin_data=plugin)
        s = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(
                tag=tag, metadata=smd, tensor=tensor)])
        self._write_event(self._event_pb2.Event(
            wall_time=time.time(), step=step, summary=s))

    def add_hparams(self, hparam_dict, metric_dict, step):
        """Log one trial as HParams (shown in TensorBoard HParams tab)."""
        # Encode as scalars under a run-specific tag prefix
        for k, v in {**hparam_dict, **metric_dict}.items():
            self.add_scalar(f"hparams/{k}", float(v), step)

    def close(self):
        self._f.close()

# ============================================================================
#  YAML helpers
# ============================================================================
def read_yaml_text(path):
    with open(path, "r") as f:
        return f.read()

def write_params_to_yaml(yaml_path, param_dict):
    lines = read_yaml_text(yaml_path).splitlines(keepends=True)
    out = []
    for line in lines:
        replaced = False
        for key, val in param_dict.items():
            stripped = line.lstrip()
            if stripped.startswith(key + ":"):
                indent = line[: len(line) - len(stripped)]
                if isinstance(val, float):
                    val_str = f"{val}" if val != int(val) else f"{val}"
                else:
                    val_str = str(val)
                out.append(f"{indent}{key}: {val_str}\n")
                replaced = True
                break
        if not replaced:
            out.append(line)
    with open(yaml_path, "w") as f:
        f.writelines(out)

# ============================================================================
#  Run preprocessor / localization
# ============================================================================
def run_preprocessor(yaml_path, timeout=600):
    try:
        r = subprocess.run(
            [sys.executable, PREPROCESSOR, yaml_path],
            capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"    [WARN] preprocessor failed: {r.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    [WARN] preprocessor timed out")
        return False

def run_localization(n_eval_frames, track="Track2", timeout=300):
    ros_result_dir = "/root/catkin_ws/src/localization/result/"
    existing = set(glob.glob(os.path.join(ros_result_dir, "track*_diag_*.txt")))

    proc = subprocess.Popen(
        ["roslaunch", "localization", "localization.launch",
         "use_rviz:=false", f"track:={track}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid)

    new_diag = None
    start = time.time()
    try:
        while time.time() - start < timeout:
            time.sleep(3)
            current = set(glob.glob(os.path.join(ros_result_dir, "track*_diag_*.txt")))
            new_files = current - existing
            if new_files:
                new_diag = max(new_files, key=os.path.getmtime)
                count = 0
                with open(new_diag) as f:
                    for i, line in enumerate(f):
                        if i > 0 and not line.startswith("#"):
                            count += 1
                if count >= n_eval_frames:
                    break
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    return new_diag

# ============================================================================
#  Parse diag → metrics
# ============================================================================
def parse_diag(filepath, n_eval_frames=50):
    rows = []
    try:
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["frame"].startswith("#"):
                    continue
                rows.append(r)
                if len(rows) >= n_eval_frames:
                    break
    except Exception as e:
        print(f"    [WARN] cannot parse {filepath}: {e}")
        return None

    if len(rows) < 5:
        print(f"    [WARN] only {len(rows)} rows")
        return None

    scores = [float(r["score"]) for r in rows]
    pts = [int(r["radar_pts"]) for r in rows]
    converged = [int(r["converged"]) for r in rows]

    mean_fitness = sum(scores) / len(scores)
    radar_pts_mean = sum(pts) / len(pts)
    failed_ratio = sum(1 for c in converged if c == 0) / len(converged)
    composite = (mean_fitness * 0.5
                 + (1000.0 / max(radar_pts_mean, 1)) * 0.3
                 + failed_ratio * 0.2)
    return {
        "mean_fitness": round(mean_fitness, 4),
        "radar_pts_mean": round(radar_pts_mean, 1),
        "failed_ratio": round(failed_ratio, 4),
        "composite_score": round(composite, 4),
        "n_frames": len(rows),
    }

# ============================================================================
#  CSV logger
# ============================================================================
CSV_HEADER = PARAM_NAMES + [
    "mean_fitness", "radar_pts_mean", "failed_ratio",
    "composite_score", "n_frames", "elapsed_sec",
]

def init_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

def append_csv(csv_path, param_dict, metrics, elapsed):
    row = [param_dict[k] for k in PARAM_NAMES]
    row += [metrics["mean_fitness"], metrics["radar_pts_mean"],
            metrics["failed_ratio"], metrics["composite_score"],
            metrics["n_frames"], round(elapsed, 1)]
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ============================================================================
#  Grid / random / neighbour generation
# ============================================================================
def full_grid():
    values_list = [SEARCH_SPACE[k] for k in PARAM_NAMES]
    return [dict(zip(PARAM_NAMES, vals))
            for vals in itertools.product(*values_list)]

def random_sample(n):
    grid = full_grid()
    random.seed(42)
    return random.sample(grid, min(n, len(grid)))

def neighbours(combo):
    nbrs = []
    for key in PARAM_NAMES:
        vals = SEARCH_SPACE[key]
        cur_idx = vals.index(combo[key]) if combo[key] in vals else -1
        for delta in [-1, 1]:
            new_idx = cur_idx + delta
            if 0 <= new_idx < len(vals):
                c = dict(combo)
                c[key] = vals[new_idx]
                nbrs.append(c)
    seen = set()
    unique = []
    for c in nbrs:
        k = tuple(c[p] for p in PARAM_NAMES)
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique

# ============================================================================
#  TensorBoard: real-time logging helpers
# ============================================================================
def tb_log_trial(tb, trial_idx, combo, metrics, elapsed, phase):
    """Log one completed trial to TensorBoard."""
    step = trial_idx

    # 1) Live scalars — watch these curves grow in real time
    tb.add_scalar("live/composite_score", metrics["composite_score"], step)
    tb.add_scalar("live/mean_fitness", metrics["mean_fitness"], step)
    tb.add_scalar("live/radar_pts_mean", metrics["radar_pts_mean"], step)
    tb.add_scalar("live/failed_ratio", metrics["failed_ratio"], step)
    tb.add_scalar("live/elapsed_sec", elapsed, step)

    # 2) Running best — monotonically decreasing
    #    (stored as attribute on the function)
    if not hasattr(tb_log_trial, "_best"):
        tb_log_trial._best = float("inf")
    tb_log_trial._best = min(tb_log_trial._best, metrics["composite_score"])
    tb.add_scalar("live/best_score_so_far", tb_log_trial._best, step)

    # 3) Per-parameter value used (for correlating in TensorBoard)
    for k in PARAM_NAMES:
        tb.add_scalar(f"params/{k}", float(combo[k]), step)

    # 4) Phase marker
    tb.add_scalar("live/phase", 1.0 if phase == "random" else 2.0, step)


def tb_log_sensitivity(tb, all_results):
    """After a phase completes, log parameter sensitivity bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, param in enumerate(PARAM_NAMES):
        ax = axes[i]
        groups = {}
        for combo, metrics in all_results:
            v = combo[param]
            groups.setdefault(v, []).append(metrics["composite_score"])

        labels = sorted(groups.keys(), key=float)
        means = [np.mean(groups[l]) for l in labels]
        mins = [np.min(groups[l]) for l in labels]
        counts = [len(groups[l]) for l in labels]

        x = np.arange(len(labels))
        bars = ax.bar(x, means, alpha=0.7, color="steelblue", label="mean")
        ax.scatter(x, mins, color="red", zorder=5, s=80, label="min")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in labels], fontsize=10)
        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_ylabel("composite score")
        ax.legend(fontsize=8)
        # Annotate count
        for j, c in enumerate(counts):
            ax.annotate(f"n={c}", (x[j], means[j]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="gray")

    plt.suptitle(f"Parameter Sensitivity ({len(all_results)} trials)", fontsize=14)
    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def tb_log_heatmaps(tb, all_results, step):
    """Log 2D parameter interaction heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [
        ("accumulate_frames", "accumulate_voxel_size"),
        ("radar_z_min", "radar_z_max"),
        ("radar_sol_mean_k", "radar_sol_std_dev_mul"),
        ("range_min", "range_max"),
    ]

    for p1, p2 in pairs:
        groups = {}
        for combo, metrics in all_results:
            key = (combo[p1], combo[p2])
            groups.setdefault(key, []).append(metrics["composite_score"])

        v1 = sorted(set(k[0] for k in groups))
        v2 = sorted(set(k[1] for k in groups))
        grid = np.full((len(v2), len(v1)), np.nan)
        for (a, b), scores in groups.items():
            grid[v2.index(b), v1.index(a)] = np.mean(scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(v1)))
        ax.set_xticklabels([f"{v}" for v in v1])
        ax.set_yticks(range(len(v2)))
        ax.set_yticklabels([f"{v}" for v in v2])
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f"Mean score: {p1} vs {p2}")
        for ii in range(len(v2)):
            for jj in range(len(v1)):
                if not np.isnan(grid[ii, jj]):
                    color = "white" if grid[ii, jj] > np.nanmedian(grid) else "black"
                    ax.text(jj, ii, f"{grid[ii,jj]:.2f}", ha="center",
                            va="center", fontsize=11, fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, label="composite score (lower=better)")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        tb.add_image(f"heatmap/{p1}_vs_{p2}", img, step)
        plt.close(fig)


def tb_log_parallel_coords(tb, all_results, step):
    """Log parallel coordinates plot as TensorBoard image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    scores = np.array([m["composite_score"] for _, m in all_results])
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.cm.RdYlGn_r

    # Normalize params to [0,1]
    data = {}
    for p in PARAM_NAMES:
        vals = np.array([c[p] for c, _ in all_results])
        vmin, vmax = vals.min(), vals.max()
        data[p] = (vals - vmin) / (vmax - vmin + 1e-9)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(PARAM_NAMES))
    for i in range(len(all_results)):
        y = [data[p][i] for p in PARAM_NAMES]
        ax.plot(x, y, color=cmap(norm(scores[i])), alpha=0.4, linewidth=1)

    short_names = [p.replace("radar_", "r_").replace("accumulate_", "acc_")
                    .replace("_sol_", "_") for p in PARAM_NAMES]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Parallel Coordinates ({len(all_results)} trials, green=better)")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="composite_score")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    tb.add_image("parallel_coordinates", img, step)
    plt.close(fig)


def tb_log_ranked(tb, all_results):
    """Log final ranked curves (sorted by score, x=rank)."""
    sorted_results = sorted(all_results, key=lambda x: x[1]["composite_score"])
    for rank, (combo, metrics) in enumerate(sorted_results, 1):
        tb.add_scalar("ranked/composite_score", metrics["composite_score"], rank)
        tb.add_scalar("ranked/mean_fitness", metrics["mean_fitness"], rank)
        tb.add_scalar("ranked/radar_pts_mean", metrics["radar_pts_mean"], rank)
        tb.add_scalar("ranked/failed_ratio", metrics["failed_ratio"], rank)


def tb_log_text_summary(tb, all_results, current_params, step):
    """Log markdown summary to TensorBoard Text tab."""
    sorted_r = sorted(all_results, key=lambda x: x[1]["composite_score"])
    lines = ["## Top 5 Best Combinations\n"]
    for i, (combo, m) in enumerate(sorted_r[:5]):
        lines.append(f"### #{i+1}  score={m['composite_score']:.4f}  "
                     f"fitness={m['mean_fitness']:.3f}  "
                     f"pts={m['radar_pts_mean']:.0f}  "
                     f"failed={m['failed_ratio']:.2%}")
        for k in PARAM_NAMES:
            cur = current_params.get(k)
            diff = f"  **(current: {cur})**" if cur is not None and float(cur) != float(combo[k]) else ""
            lines.append(f"  {k}: **{combo[k]}**{diff}")
        lines.append("")

    lines.append("## Best Value per Parameter\n")
    lines.append("| Parameter | Best Val | Mean Score | Current | Match |")
    lines.append("|-----------|----------|------------|---------|-------|")
    for param in PARAM_NAMES:
        groups = {}
        for c, m in all_results:
            groups.setdefault(c[param], []).append(m["composite_score"])
        best_v = min(groups, key=lambda v: np.mean(groups[v]))
        cur = current_params.get(param, "?")
        match = "OK" if float(cur) == float(best_v) else "DIFF"
        lines.append(f"| {param} | {best_v} | {np.mean(groups[best_v]):.3f} | {cur} | {match} |")

    tb.add_text("summary", "\n".join(lines), step)


def tb_log_plotly_html(all_results, output_html):
    """Save interactive Plotly parallel coordinates HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly not installed, skipping HTML export")
        return

    scores = [m["composite_score"] for _, m in all_results]
    dims = []
    for p in PARAM_NAMES:
        vals = [c[p] for c, _ in all_results]
        dims.append(dict(
            label=p.replace("radar_", "r_").replace("accumulate_", "acc_")
                   .replace("_sol_", "_"),
            values=vals, range=[min(vals), max(vals)]))
    for metric in ["mean_fitness", "radar_pts_mean", "composite_score"]:
        vals = [m[metric] for _, m in all_results]
        dims.append(dict(label=metric, values=vals, range=[min(vals), max(vals)]))

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=scores,
                  colorscale=[[0, "green"], [0.5, "gold"], [1, "red"]],
                  showscale=True, cmin=min(scores), cmax=max(scores),
                  colorbar=dict(title="Score")),
        dimensions=dims))
    fig.update_layout(title="Grid Search Parallel Coordinates (green=better)",
                      font=dict(size=11), height=700,
                      margin=dict(l=80, r=80, t=60, b=30))
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"  Plotly HTML saved: {output_html}")

# ============================================================================
#  Evaluate one combo
# ============================================================================
def evaluate_combo(combo, yaml_path, n_eval_frames, csv_path, track="Track2"):
    t0 = time.time()
    write_params_to_yaml(yaml_path, combo)

    if not run_preprocessor(yaml_path):
        return None, 0

    diag_path = run_localization(n_eval_frames, track=track)
    if diag_path is None:
        print("    [FAIL] no diag file produced")
        return None, 0

    metrics = parse_diag(diag_path, n_eval_frames)
    elapsed = time.time() - t0

    if metrics:
        append_csv(csv_path, combo, metrics, elapsed)
        print(f"    score={metrics['composite_score']:.3f}  "
              f"fitness={metrics['mean_fitness']:.3f}  "
              f"pts={metrics['radar_pts_mean']:.0f}  "
              f"failed={metrics['failed_ratio']:.2%}  "
              f"({elapsed:.0f}s)")
    return metrics, elapsed

# ============================================================================
#  Print / apply results
# ============================================================================
def print_top_results(all_results, n=5):
    sorted_r = sorted(all_results, key=lambda x: x[1]["composite_score"])
    print(f"\n{'='*80}")
    print(f"  Top {n} parameter combinations")
    print(f"{'='*80}")
    for i, (combo, m) in enumerate(sorted_r[:n]):
        print(f"\n  #{i+1}  composite={m['composite_score']:.4f}  "
              f"fitness={m['mean_fitness']:.3f}  "
              f"pts={m['radar_pts_mean']:.0f}  "
              f"failed={m['failed_ratio']:.2%}")
        for k in PARAM_NAMES:
            print(f"       {k}: {combo[k]}")

def apply_best(all_results, yaml_path):
    if not all_results:
        return None
    combo, metrics = min(all_results, key=lambda x: x[1]["composite_score"])
    out = {}
    for k in PARAM_NAMES:
        if k in ("accumulate_frames", "radar_sol_mean_k"):
            out[k] = int(float(combo[k]))
        else:
            out[k] = float(combo[k])
    write_params_to_yaml(yaml_path, out)
    return out

def read_current_params():
    import yaml
    with open(PARAMS_YAML) as f:
        params = yaml.safe_load(f)
    return {k: float(params[k]) for k in PARAM_NAMES if k in params}

# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Grid search for radar preprocessor params (TensorBoard)")
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--n-eval-frames", type=int, default=50)
    parser.add_argument("--track", type=str, default="Track2")
    parser.add_argument("--fine-top-k", type=int, default=10)
    parser.add_argument("--skip-phase1", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULT_DIR, f"grid_search_result_{timestamp}.csv")
    html_path = os.path.join(RESULT_DIR, "parallel_coords.html")

    # Backup
    backup_path = PARAMS_YAML + ".bak"
    shutil.copy2(PARAMS_YAML, backup_path)
    print(f"Backup: {backup_path}")

    current_params = read_current_params()

    # Init TensorBoard + CSV
    tb = TBWriter(RUNS_DIR)
    init_csv(csv_path)
    all_results = []   # list of (combo, metrics)
    all_elapsed = []
    trial_idx = 0

    print(f"\nTensorBoard logs: {RUNS_DIR}")
    print(f"  tensorboard --logdir={RUNS_DIR} --port=6006 --bind_all\n")

    # ── Phase 1: Random search ──────────────────────────────────
    if not args.skip_phase1:
        combos = random_sample(args.n_random)
        print(f"{'='*60}")
        print(f"  Phase 1: Random Search — {len(combos)} combos")
        print(f"  Eval frames: {args.n_eval_frames}")
        print(f"{'='*60}\n")

        for i, combo in enumerate(combos):
            if all_elapsed:
                avg_t = sum(all_elapsed) / len(all_elapsed)
                eta = avg_t * (len(combos) - i) / 60
                print(f"[{i+1}/{len(combos)}] ETA: {eta:.1f} min")
            else:
                print(f"[{i+1}/{len(combos)}]")

            short = ", ".join(f"{k.split('_')[-1]}={combo[k]}" for k in PARAM_NAMES)
            print(f"    {short}")

            metrics, elapsed = evaluate_combo(
                combo, PARAMS_YAML, args.n_eval_frames, csv_path, args.track)
            all_elapsed.append(elapsed)

            if metrics:
                all_results.append((combo, metrics))
                tb_log_trial(tb, trial_idx, combo, metrics, elapsed, "random")
                trial_idx += 1

        # End of Phase 1: log visualizations
        if all_results:
            print("\n  Logging Phase 1 visualizations to TensorBoard...")
            img = tb_log_sensitivity(tb, all_results)
            tb.add_image("sensitivity/phase1", img, 1)
            tb_log_heatmaps(tb, all_results, 1)
            tb_log_parallel_coords(tb, all_results, 1)
            tb_log_ranked(tb, all_results)
            tb_log_text_summary(tb, all_results, current_params, 1)

    # ── Phase 2: Fine search ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine Search around top-{args.fine_top_k}")
    print(f"{'='*60}\n")

    sorted_results = sorted(all_results, key=lambda x: x[1]["composite_score"])
    tested = set()
    for combo, _ in all_results:
        tested.add(tuple(combo[p] for p in PARAM_NAMES))

    fine_combos = []
    for combo, _ in sorted_results[:args.fine_top_k]:
        typed = {}
        for k in PARAM_NAMES:
            if k in ("accumulate_frames", "radar_sol_mean_k"):
                typed[k] = int(float(combo[k]))
            else:
                typed[k] = float(combo[k])
        for nbr in neighbours(typed):
            key = tuple(nbr[p] for p in PARAM_NAMES)
            if key not in tested:
                tested.add(key)
                fine_combos.append(nbr)

    print(f"  Fine combos: {len(fine_combos)}\n")

    for i, combo in enumerate(fine_combos):
        if all_elapsed:
            eta = (sum(all_elapsed) / len(all_elapsed)) * (len(fine_combos) - i) / 60
            print(f"[Fine {i+1}/{len(fine_combos)}] ETA: {eta:.1f} min")
        else:
            print(f"[Fine {i+1}/{len(fine_combos)}]")

        short = ", ".join(f"{k.split('_')[-1]}={combo[k]}" for k in PARAM_NAMES)
        print(f"    {short}")

        metrics, elapsed = evaluate_combo(
            combo, PARAMS_YAML, args.n_eval_frames, csv_path, args.track)
        all_elapsed.append(elapsed)

        if metrics:
            all_results.append((combo, metrics))
            tb_log_trial(tb, trial_idx, combo, metrics, elapsed, "fine")
            trial_idx += 1

    # ── Final summary ───────────────────────────────────────────
    if all_results:
        print("\n  Logging final visualizations...")
        img = tb_log_sensitivity(tb, all_results)
        tb.add_image("sensitivity/final", img, 2)
        tb_log_heatmaps(tb, all_results, 2)
        tb_log_parallel_coords(tb, all_results, 2)
        tb_log_ranked(tb, all_results)
        tb_log_text_summary(tb, all_results, current_params, 2)
        tb_log_plotly_html(all_results, html_path)

    tb.close()

    # Print & apply
    print_top_results(all_results, n=5)

    best = apply_best(all_results, PARAMS_YAML)
    if best:
        print(f"\nBest combo written to {PARAMS_YAML}:")
        for k, v in best.items():
            print(f"  {k}: {v}")

    total_time = sum(all_elapsed)
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Results CSV: {csv_path}")
    print(f"TensorBoard: tensorboard --logdir={RUNS_DIR} --port=6006 --bind_all")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
