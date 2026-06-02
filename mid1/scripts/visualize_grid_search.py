#!/usr/bin/env python3
"""
Visualize grid search results with TensorBoard + Plotly.

Usage:
  python3 scripts/visualize_grid_search.py [--csv result/grid_search_result_*.csv]

Then:
  tensorboard --logdir=runs/grid_search --port=6006

Also produces:
  result/parallel_coords.html   (standalone, no TensorBoard needed)
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
import yaml

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
PARAMS_YAML = os.path.join(PKG_DIR, "config", "params.yaml")
RESULT_DIR = os.path.join(PKG_DIR, "result")
RUNS_DIR = os.path.join(PKG_DIR, "runs", "grid_search")

PARAM_NAMES = [
    "radar_z_min", "radar_z_max", "range_min", "range_max",
    "accumulate_frames", "accumulate_voxel_size",
    "radar_sol_mean_k", "radar_sol_std_dev_mul",
]
METRIC_NAMES = ["mean_fitness", "radar_pts_mean", "failed_ratio", "composite_score"]

# ---------------------------------------------------------------------------
#  Read current params.yaml
# ---------------------------------------------------------------------------
def read_current_params():
    with open(PARAMS_YAML) as f:
        params = yaml.safe_load(f)
    return {k: float(params[k]) for k in PARAM_NAMES if k in params}

# ---------------------------------------------------------------------------
#  Load CSV
# ---------------------------------------------------------------------------
def load_csv(csv_path):
    """Returns list of dicts, sorted by composite_score ascending."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Skip empty rows
            if not r.get("composite_score"):
                continue
            row = {}
            for k in PARAM_NAMES:
                row[k] = float(r[k])
            for k in METRIC_NAMES:
                row[k] = float(r[k])
            if "n_frames" in r and r["n_frames"]:
                row["n_frames"] = int(r["n_frames"])
            if "elapsed_sec" in r and r["elapsed_sec"]:
                row["elapsed_sec"] = float(r["elapsed_sec"])
            rows.append(row)
    rows.sort(key=lambda x: x["composite_score"])
    return rows

# ---------------------------------------------------------------------------
#  Generate demo data (when CSV is empty)
# ---------------------------------------------------------------------------
def generate_demo_data(n=100):
    """Produce synthetic grid search results for visualization testing."""
    import itertools, random
    random.seed(42)
    np.random.seed(42)

    search_space = {
        "radar_z_min":          [-8.0, -6.0, -4.0],
        "radar_z_max":          [15.0, 20.0, 25.0],
        "range_min":            [1.5, 2.0, 3.0],
        "range_max":            [50.0, 60.0, 80.0],
        "accumulate_frames":    [3.0, 5.0, 8.0],
        "accumulate_voxel_size": [0.3, 0.5, 0.7],
        "radar_sol_mean_k":     [10.0, 20.0, 30.0],
        "radar_sol_std_dev_mul": [1.5, 2.0, 2.5],
    }
    all_combos = list(itertools.product(*[search_space[k] for k in PARAM_NAMES]))
    sampled = random.sample(all_combos, min(n, len(all_combos)))

    rows = []
    for vals in sampled:
        combo = dict(zip(PARAM_NAMES, [float(v) for v in vals]))
        # Synthetic fitness: prefer mid-range z, moderate range_max, etc.
        fitness = (
            2.0
            + 0.3 * abs(combo["radar_z_min"] + 6.0)
            + 0.2 * abs(combo["radar_z_max"] - 20.0) / 5.0
            + 0.5 * (combo["range_max"] - 50.0) / 30.0
            + 0.3 * abs(combo["accumulate_frames"] - 5.0) / 3.0
            + 0.2 * abs(combo["accumulate_voxel_size"] - 0.5) / 0.2
            + 0.1 * abs(combo["radar_sol_mean_k"] - 20.0) / 10.0
            + 0.1 * abs(combo["radar_sol_std_dev_mul"] - 2.0) / 0.5
            + np.random.normal(0, 0.3)
        )
        fitness = max(0.5, fitness)
        pts = max(50, 300 - fitness * 30 + np.random.normal(0, 20))
        failed = max(0, min(1, (fitness - 2.0) * 0.05 + np.random.normal(0, 0.01)))
        score = fitness * 0.5 + (1000.0 / max(pts, 1)) * 0.3 + failed * 0.2

        combo["mean_fitness"] = round(fitness, 4)
        combo["radar_pts_mean"] = round(pts, 1)
        combo["failed_ratio"] = round(failed, 4)
        combo["composite_score"] = round(score, 4)
        combo["n_frames"] = 50
        rows.append(combo)

    rows.sort(key=lambda x: x["composite_score"])
    return rows

# ---------------------------------------------------------------------------
#  1. TensorBoard Scalars
# ---------------------------------------------------------------------------
def write_scalars(writer, rows, current_params):
    """Write ranked scalar curves + mark current params position."""
    # Find rank of current params (closest match)
    current_rank = None
    best_dist = float("inf")
    for i, row in enumerate(rows):
        dist = sum((row[k] - current_params.get(k, 0)) ** 2 for k in PARAM_NAMES)
        if dist < best_dist:
            best_dist = dist
            current_rank = i

    for i, row in enumerate(rows):
        rank = i + 1
        writer.add_scalar("ranked/mean_fitness", row["mean_fitness"], rank)
        writer.add_scalar("ranked/radar_pts_mean", row["radar_pts_mean"], rank)
        writer.add_scalar("ranked/failed_ratio", row["failed_ratio"], rank)
        writer.add_scalar("ranked/composite_score", row["composite_score"], rank)

    # Mark current params with a vertical indicator
    if current_rank is not None:
        r = rows[current_rank]
        rank = current_rank + 1
        writer.add_scalar("current_params/mean_fitness", r["mean_fitness"], rank)
        writer.add_scalar("current_params/radar_pts_mean", r["radar_pts_mean"], rank)
        writer.add_scalar("current_params/composite_score", r["composite_score"], rank)
        # Also add points at rank 0 and max to make marker line visible
        writer.add_scalar("current_params/composite_score", r["composite_score"], 0)
        writer.add_scalar("current_params/composite_score", r["composite_score"], len(rows))

    # Per-parameter sensitivity: for each param value, mean composite_score
    for param in PARAM_NAMES:
        groups = {}
        for row in rows:
            v = row[param]
            groups.setdefault(v, []).append(row["composite_score"])
        for v in sorted(groups.keys()):
            mean_s = sum(groups[v]) / len(groups[v])
            min_s = min(groups[v])
            # Use the param value * 100 as step to spread points on x-axis
            step = int(v * 100)
            writer.add_scalar(f"sensitivity_mean/{param}", mean_s, step)
            writer.add_scalar(f"sensitivity_min/{param}", min_s, step)

# ---------------------------------------------------------------------------
#  2. Parallel Coordinates (Plotly)
# ---------------------------------------------------------------------------
def make_parallel_coords(rows, output_html):
    import plotly.graph_objects as go

    scores = [r["composite_score"] for r in rows]
    score_min, score_max = min(scores), max(scores)

    dimensions = []
    for p in PARAM_NAMES:
        vals = [r[p] for r in rows]
        dimensions.append(dict(
            label=p.replace("radar_", "r_").replace("accumulate_", "acc_")
                   .replace("_sol_", "_"),
            values=vals,
            range=[min(vals), max(vals)],
        ))
    # Add metrics
    for m in ["mean_fitness", "radar_pts_mean", "composite_score"]:
        vals = [r[m] for r in rows]
        dimensions.append(dict(label=m, values=vals, range=[min(vals), max(vals)]))

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=scores,
            colorscale=[[0, "green"], [0.5, "gold"], [1, "red"]],
            showscale=True,
            cmin=score_min,
            cmax=score_max,
            colorbar=dict(title="Score"),
        ),
        dimensions=dimensions,
    ))
    fig.update_layout(
        title="Grid Search: Parallel Coordinates (green=better)",
        font=dict(size=11),
        height=700,
        margin=dict(l=80, r=80, t=60, b=30),
    )
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"Parallel coordinates saved: {output_html}")

# ---------------------------------------------------------------------------
#  2b. Parallel Coordinates as TensorBoard image
# ---------------------------------------------------------------------------
def write_parallel_coords_image(writer, rows):
    """Render a matplotlib approximation for TensorBoard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    scores = np.array([r["composite_score"] for r in rows])
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.cm.RdYlGn_r  # green=low, red=high

    # Normalize all param values to [0,1] for display
    data = {}
    for p in PARAM_NAMES:
        vals = np.array([r[p] for r in rows])
        vmin, vmax = vals.min(), vals.max()
        data[p] = (vals - vmin) / (vmax - vmin + 1e-9)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(PARAM_NAMES))

    for i, row in enumerate(rows):
        y = [data[p][i] for p in PARAM_NAMES]
        ax.plot(x, y, color=cmap(norm(scores[i])), alpha=0.4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [p.replace("radar_", "r_").replace("accumulate_", "acc_")
          .replace("_sol_", "_") for p in PARAM_NAMES],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel Coordinates (green=low score=better)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="composite_score")

    fig.tight_layout()
    fig.canvas.draw()

    # Convert to numpy array for TensorBoard
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    writer.add_image("parallel_coordinates", img, 0, dataformats="HWC")
    plt.close(fig)

# ---------------------------------------------------------------------------
#  3. Heatmaps
# ---------------------------------------------------------------------------
def write_heatmaps(writer, rows):
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
        # Group by (p1, p2) → mean score
        groups = {}
        for r in rows:
            key = (r[p1], r[p2])
            groups.setdefault(key, []).append(r["composite_score"])

        v1_vals = sorted(set(k[0] for k in groups))
        v2_vals = sorted(set(k[1] for k in groups))

        grid = np.full((len(v2_vals), len(v1_vals)), np.nan)
        for (a, b), scores in groups.items():
            i = v2_vals.index(b)
            j = v1_vals.index(a)
            grid[i, j] = np.mean(scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(v1_vals)))
        ax.set_xticklabels([f"{v}" for v in v1_vals])
        ax.set_yticks(range(len(v2_vals)))
        ax.set_yticklabels([f"{v}" for v in v2_vals])
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f"Mean composite_score: {p1} vs {p2}")

        # Annotate cells
        for i in range(len(v2_vals)):
            for j in range(len(v1_vals)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if grid[i, j] > np.nanmedian(grid) else "black")

        fig.colorbar(im, ax=ax, label="composite_score (lower=better)")
        fig.tight_layout()
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        tag = f"heatmap/{p1}_vs_{p2}"
        writer.add_image(tag, img, 0, dataformats="HWC")
        plt.close(fig)

# ---------------------------------------------------------------------------
#  4. Text summary
# ---------------------------------------------------------------------------
def write_text_summary(writer, rows, current_params):
    lines = []

    # Top 5
    lines.append("## Top 5 Best Combinations\n")
    for i, r in enumerate(rows[:5]):
        lines.append(f"### #{i+1}  composite={r['composite_score']:.4f}  "
                     f"fitness={r['mean_fitness']:.3f}  "
                     f"pts={r['radar_pts_mean']:.0f}  "
                     f"failed={r['failed_ratio']:.2%}")
        for k in PARAM_NAMES:
            marker = ""
            if k in current_params and r[k] != current_params[k]:
                marker = f"  (current: {current_params[k]})"
            lines.append(f"  {k}: **{r[k]}**{marker}")
        lines.append("")

    # Best range per parameter
    lines.append("## Best Range per Parameter\n")
    lines.append("| Parameter | Best Value(s) | Mean Score | Current | Match? |")
    lines.append("|-----------|--------------|------------|---------|--------|")
    for param in PARAM_NAMES:
        groups = {}
        for r in rows:
            v = r[param]
            groups.setdefault(v, []).append(r["composite_score"])
        best_v = min(groups, key=lambda v: np.mean(groups[v]))
        best_score = np.mean(groups[best_v])
        cur = current_params.get(param, "?")
        match = "OK" if float(cur) == float(best_v) else "DIFF"
        lines.append(f"| {param} | {best_v} | {best_score:.3f} | {cur} | {match} |")

    # Gap analysis
    lines.append("\n## Current Params vs Best Combo\n")
    best = rows[0]
    lines.append("| Parameter | Current | Best | Delta |")
    lines.append("|-----------|---------|------|-------|")
    for k in PARAM_NAMES:
        cur = current_params.get(k, 0)
        bst = best[k]
        delta = bst - cur
        lines.append(f"| {k} | {cur} | {bst} | {delta:+.1f} |")

    text = "\n".join(lines)
    writer.add_text("summary", text, 0)
    # Also print to console
    print("\n" + text)

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize grid search results")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to grid_search_result CSV. Auto-detects newest if omitted.")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data (for testing visualization)")
    args = parser.parse_args()

    # Find CSV
    if args.csv:
        csv_path = args.csv
    else:
        candidates = sorted(glob.glob(os.path.join(RESULT_DIR, "grid_search_result*.csv")))
        csv_path = candidates[-1] if candidates else None

    # Load data
    use_demo = args.demo
    if csv_path and os.path.exists(csv_path):
        rows = load_csv(csv_path)
        if len(rows) < 3:
            print(f"[WARN] CSV has only {len(rows)} rows, using demo data")
            use_demo = True
        else:
            print(f"Loaded {len(rows)} results from {csv_path}")
    else:
        print("[WARN] No CSV found, using demo data")
        use_demo = True

    if use_demo:
        rows = generate_demo_data(100)
        print(f"Generated {len(rows)} demo combos for visualization testing")

    # Read current params
    current_params = read_current_params()
    print(f"Current params.yaml: { {k: current_params[k] for k in PARAM_NAMES if k in current_params} }")

    # Setup TensorBoard writer
    os.makedirs(RUNS_DIR, exist_ok=True)
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.compat.proto import event_pb2
    # Use a thin wrapper that doesn't need torch
    from tensorboard import program as _  # ensure tb is importable

    # We'll use a minimal writer
    writer = TBWriter(RUNS_DIR)

    print("\nWriting TensorBoard data...")

    # 1. Scalars
    write_scalars(writer, rows, current_params)

    # 2. Parallel coordinates
    write_parallel_coords_image(writer, rows)

    # 3. Heatmaps
    write_heatmaps(writer, rows)

    # 4. Text summary
    write_text_summary(writer, rows, current_params)

    writer.close()
    print(f"\nTensorBoard logs written to: {RUNS_DIR}")

    # 5. Plotly parallel coordinates
    html_path = os.path.join(RESULT_DIR, "parallel_coords.html")
    make_parallel_coords(rows, html_path)

    # Print launch command
    tb_bin = os.path.expanduser("~/.local/bin/tensorboard")
    if not os.path.exists(tb_bin):
        tb_bin = "tensorboard"
    print(f"\nLaunch TensorBoard:")
    print(f"  {tb_bin} --logdir={RUNS_DIR} --port=6006 --bind_all")
    print(f"\nOr open directly: {html_path}")


# ---------------------------------------------------------------------------
#  Minimal TensorBoard writer (no torch dependency)
# ---------------------------------------------------------------------------
class TBWriter:
    """Lightweight TensorBoard writer using tensorboard's own protobuf."""

    def __init__(self, logdir):
        import struct
        from tensorboard.compat.proto import event_pb2, summary_pb2
        self._logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self._filename = os.path.join(
            logdir, f"events.out.tfevents.{int(__import__('time').time())}.grid_search"
        )
        self._writer = open(self._filename, "wb")
        self._event_pb2 = event_pb2
        self._summary_pb2 = summary_pb2
        # Write file_version event
        event = event_pb2.Event(file_version="brain.Event:2")
        self._write_event(event)

    def _write_event(self, event):
        import struct, hashlib
        serialized = event.SerializeToString()
        # TFRecord format: uint64 length, uint32 crc, data, uint32 crc
        header = struct.pack("Q", len(serialized))
        header_crc = struct.pack("I", self._masked_crc32c(header))
        data_crc = struct.pack("I", self._masked_crc32c(serialized))
        self._writer.write(header + header_crc + serialized + data_crc)

    @staticmethod
    def _masked_crc32c(data):
        """CRC32C with TensorFlow masking."""
        try:
            import crcmod
            crc_fn = crcmod.predefined.mkCrcFun("crc-32c")
            crc = crc_fn(data)
        except ImportError:
            # Fallback: simple crc32 (not crc32c, but works for most cases)
            import binascii
            crc = binascii.crc32(data) & 0xFFFFFFFF
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    def add_scalar(self, tag, value, step):
        summary = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
        )
        event = self._event_pb2.Event(
            wall_time=__import__("time").time(),
            step=step,
            summary=summary,
        )
        self._write_event(event)

    def add_image(self, tag, img_array, step, dataformats="HWC"):
        """Write image from numpy array (H, W, 3)."""
        import io
        from PIL import Image
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        h, w = img_array.shape[:2]
        image_proto = self._summary_pb2.Summary.Image(
            height=h, width=w, colorspace=3,
            encoded_image_string=png_bytes,
        )
        summary = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(tag=tag, image=image_proto)]
        )
        event = self._event_pb2.Event(
            wall_time=__import__("time").time(), step=step, summary=summary,
        )
        self._write_event(event)

    def add_text(self, tag, text, step):
        """Write text as markdown via tensor proto."""
        # TensorBoard text plugin expects a string_val in a tensor
        from tensorboard.compat.proto import tensor_pb2, tensor_shape_pb2, types_pb2

        tensor = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            string_val=[text.encode("utf-8")],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[]),
        )
        plugin_data = self._summary_pb2.SummaryMetadata.PluginData(
            plugin_name="text",
        )
        smd = self._summary_pb2.SummaryMetadata(plugin_data=plugin_data)
        summary = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(
                tag=tag, metadata=smd, tensor=tensor,
            )]
        )
        event = self._event_pb2.Event(
            wall_time=__import__("time").time(), step=step, summary=summary,
        )
        self._write_event(event)

    def close(self):
        self._writer.close()


if __name__ == "__main__":
    main()
