#!/bin/bash
# ============================================================
# Ablation study — full matrix
# Usage: cd starter_code && bash run_ablation.sh
# ============================================================

DATA_ROOT="../public_dev"
OUTPUT_ROOT="../outputs/ablation"
EVAL="evaluate_tracking.py"
GT_SEQ1="$DATA_ROOT/seq_1/gt_answer_seq1.csv"
GT_SEQ3="$DATA_ROOT/seq_3/gt_answer_seq3.csv"

# ── resolve starter_code absolute path ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_ROOT"

SEP="============================================================"
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m';  BOLD='\033[1m';   NC='\033[0m'

# ── per-run metric printer ─────────────────────────────────
print_eval() {
    local csv=$1 label=$2
    echo ""
    echo -e "${CYAN}  ┌─ $label ───────────────────────────────${NC}"
    python3 - "$csv" "$label" << 'PYEOF'
import sys, csv as _csv

path = sys.argv[1]
with open(path) as f:
    rows = list(_csv.DictReader(f))
if not rows:
    print("  ERROR: empty csv"); sys.exit(1)
r = rows[0]

def fmt(k, v):
    try: fv = float(v)
    except: return f"  │  {k:<28} {v}"
    if k == "MOTA_proxy":
        col = "\033[0;32m" if fv>=0.95 else ("\033[1;33m" if fv>=0.7 else "\033[0;31m")
        filled = int(fv*20)
        bar = "[" + "█"*filled + "░"*(20-filled) + f"] {fv:.4f}"
        return f"  │  {k:<28}\033[0m{col}{bar}\033[0m"
    elif k in ("FP_points","FN_points","IDSW_proxy","fragmentation_proxy"):
        col = "\033[0;32m" if fv==0 else ("\033[1;33m" if fv<50 else "\033[0;31m")
        return f"  │  {k:<28}{col}{int(fv):>10}\033[0m"
    return f"  │  {k:<28}{fv:>10.4f}"

for k,v in r.items():
    print(fmt(k,v))
print("  └" + "─"*48)
PYEOF
}

# ── core runner ────────────────────────────────────────────
# run_and_eval LABEL SEQ GT SCRIPT [extra args...]
run_and_eval() {
    local label=$1 seq=$2 gt=$3 script=$4
    shift 4
    local out_dir="$OUTPUT_ROOT/$label"
    mkdir -p "$out_dir"

    echo ""
    echo -e "${YELLOW}>>> [$label] running seq=$seq ...${NC}"

    # Run with PYTHONPATH pointing to starter_code so all local modules resolve
    PYTHONPATH="$SCRIPT_DIR" python3 "$script" \
        --data-root "$DATA_ROOT" \
        --seq "$seq" \
        --output-root "$out_dir" \
        --max-age 8 \
        --min-hits 1 \
        --max-distance 5.0 \
        --plot-every 1 \
        "$@"

    local pred_csv
    pred_csv=$(find "$out_dir" -name "result.csv" -not -path "*/__pycache__/*" | head -1)
    if [ -z "$pred_csv" ]; then
        echo -e "${RED}  ERROR: result.csv not found for [$label]${NC}"; return 1
    fi

    PYTHONPATH="$SCRIPT_DIR" python3 "$EVAL" \
        --gt "$gt" --pred "$pred_csv" \
        --out "$OUTPUT_ROOT/${label}_score.csv"

    print_eval "$OUTPUT_ROOT/${label}_score.csv" "$label"

    local vis_dir
    vis_dir=$(find "$out_dir" -type d -name "tracking_vis" | head -1)
    if [ -n "$vis_dir" ]; then
        local n; n=$(find "$vis_dir" -name "*.png" | wc -l)
        echo -e "  ${GREEN}vis → $vis_dir  ($n frames)${NC}"
    fi
}

# ══════════════════════════════════════════════════════════
# Patch helpers — all write to /tmp, print path to stdout
# stderr goes to terminal so the path assignment stays clean
# ══════════════════════════════════════════════════════════

make_patch_no_offline() {
    local dst=/tmp/main_no_offline.py
    sed \
        -e 's/^ENABLE_OFFLINE_STITCHING = True/ENABLE_OFFLINE_STITCHING = False/' \
        -e 's/^ENABLE_SWAP_CORRECTION   = True/ENABLE_SWAP_CORRECTION   = False/' \
        -e 's/^ENABLE_SWAP_CORRECTION = True/ENABLE_SWAP_CORRECTION = False/' \
        "$SCRIPT_DIR/main.py" > "$dst"
    echo "$dst"
}

make_patch_no_ego() {
    local dst=/tmp/main_no_ego.py
    sed 's/ego_delta = ego_motion\[frame_idx - 1\]/ego_delta = None  # ABLATION: ego comp disabled/' \
        "$SCRIPT_DIR/main.py" > "$dst"
    echo "$dst"
}

make_patch_kf_predict() {
    local tracker_src="$SCRIPT_DIR/sdc_tracker_track1.py"
    local tracker_dst=/tmp/sdc_tracker_track1_kfpred.py
    local main_dst=/tmp/main_kf_predict.py

    cp "$tracker_src" "$tracker_dst"

    # patch tracker: use kf.predict() in build_cost_matrix (stderr so stdout stays clean)
    python3 - "$tracker_dst" 1>&2 << 'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = ("    def build_cost_matrix(self, tracks, detections: np.ndarray):\n"
       "        track_xy = np.array(\n"
       "            [t[\"mean\"][:2] for t in tracks],\n"
       "            dtype=float,\n"
       "        )")
new = ("    def build_cost_matrix(self, tracks, detections: np.ndarray):\n"
       "        # ABLATION: use KF predict when available\n"
       "        predicted = []\n"
       "        for t in tracks:\n"
       "            if \"kf\" in t:\n"
       "                pred = t[\"kf\"].predict()\n"
       "            else:\n"
       "                pred = t[\"mean\"][:2]\n"
       "            predicted.append(pred)\n"
       "        track_xy = np.array(predicted, dtype=float)")

if old in src:
    src = src.replace(old, new)
    # re-enable kf in new track creation
    src = src.replace('            # "kf": kf,', '            "kf": kf,')
    with open(path, "w") as f:
        f.write(src)
    print("  patch_kf_predict: OK", file=sys.stderr)
else:
    print("  patch_kf_predict: WARNING — target not found, using unpatched copy", file=sys.stderr)
PYEOF

    # build a main that loads the patched tracker from /tmp
    # use Python so we can preserve indentation of the original import line
    python3 - "$SCRIPT_DIR/main.py" "$tracker_dst" "$main_dst" 1>&2 << 'PYEOF'
import sys, re
src_path, tracker_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src_path) as f:
    lines = f.readlines()
out = []
for line in lines:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith("from sdc_tracker_track1 import Tracker"):
        out.append(f"{indent}import importlib.util as _ilu\n")
        out.append(f"{indent}_spec = _ilu.spec_from_file_location('sdc_tracker_track1', r'{tracker_path}')\n")
        out.append(f"{indent}_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)\n")
        out.append(f"{indent}Tracker = _mod.Tracker\n")
    else:
        out.append(line)
with open(dst_path, "w") as f:
    f.writelines(out)
print(f"  main_kf_predict written to {dst_path}", file=sys.stderr)
PYEOF

    echo "$main_dst"
}

make_patch_with_kf() {
    # find the actual seq3 tracker filename
    local tracker_src=""
    for candidate in sdc_tracker_track2.py sdc_tracker.py; do
        if [ -f "$SCRIPT_DIR/$candidate" ]; then
            tracker_src="$SCRIPT_DIR/$candidate"
            break
        fi
    done

    if [ -z "$tracker_src" ]; then
        echo -e "${RED}  ERROR: cannot find seq_3 tracker (tried sdc_tracker_track2.py, sdc_tracker.py)${NC}" >&2
        echo ""
        return 1
    fi

    local tracker_dst=/tmp/sdc_tracker_withkf.py
    local main_dst=/tmp/main_with_kf.py

    cp "$tracker_src" "$tracker_dst"

    python3 - "$tracker_dst" 1>&2 << 'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# track2 already has if "kf" in track: predict() logic in match_tracks.
# We only need to enable KF initialisation when a new track is created.
# The new-track block has "kf": kf commented out — uncomment it.
old = (
    "            kf = KalmanFilter()\n"
    "            kf.x[:2] = detection[:2].reshape(2, 1)\n"
    "            self.tracks.append(\n"
    "                {\n"
    "                    \"mean\": detection[:2],\n"
    "                    \"kf\": kf,"
)
if old in src:
    print("  patch_with_kf: kf already enabled in new-track init — no patch needed", file=sys.stderr)
else:
    # try to uncomment the kf line
    old2 = '                    # "kf": kf,'
    new2 = '                    "kf": kf,'
    if old2 in src:
        src = src.replace(old2, new2, 1)
        with open(path, "w") as f:
            f.write(src)
        print("  patch_with_kf: OK — uncommented kf init in new-track block", file=sys.stderr)
    else:
        print("  patch_with_kf: WARNING — could not find kf init line to patch", file=sys.stderr)
PYEOF

    python3 - "$SCRIPT_DIR/main.py" "$tracker_dst" "$main_dst" 1>&2 << 'PYEOF'
import sys
src_path, tracker_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src_path) as f:
    lines = f.readlines()
out = []
replaced = False
for line in lines:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if not replaced and stripped.startswith("from sdc_tracker import Tracker"):
        out.append(f"{indent}import importlib.util as _ilu\n")
        out.append(f"{indent}_spec = _ilu.spec_from_file_location('sdc_tracker', r'{tracker_path}')\n")
        out.append(f"{indent}_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)\n")
        out.append(f"{indent}Tracker = _mod.Tracker\n")
        replaced = True
    elif not replaced and stripped.startswith("from sdc_tracker_track2 import Tracker"):
        out.append(f"{indent}import importlib.util as _ilu\n")
        out.append(f"{indent}_spec = _ilu.spec_from_file_location('sdc_tracker_track2', r'{tracker_path}')\n")
        out.append(f"{indent}_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)\n")
        out.append(f"{indent}Tracker = _mod.Tracker\n")
        replaced = True
    else:
        out.append(line)
with open(dst_path, "w") as f:
    f.writelines(out)
print(f"  main_with_kf written to {dst_path} (replaced={replaced})", file=sys.stderr)
PYEOF

    echo "$main_dst"
}

# ══════════════════════════════════════════════════════════
# SEQ_1
# ══════════════════════════════════════════════════════════
echo ""; echo -e "${SEP}"; echo -e "${GREEN}SEQ_1 ABLATIONS${NC}"; echo -e "${SEP}"

run_and_eval "seq1_full"       seq_1 "$GT_SEQ1" "$SCRIPT_DIR/main.py"

PATCH=$(make_patch_no_offline)
[ -n "$PATCH" ] && run_and_eval "seq1_no_offline" seq_1 "$GT_SEQ1" "$PATCH"

PATCH=$(make_patch_kf_predict)
[ -n "$PATCH" ] && run_and_eval "seq1_kf_predict" seq_1 "$GT_SEQ1" "$PATCH"

# ══════════════════════════════════════════════════════════
# SEQ_3
# ══════════════════════════════════════════════════════════
echo ""; echo -e "${SEP}"; echo -e "${GREEN}SEQ_3 ABLATIONS${NC}"; echo -e "${SEP}"

run_and_eval "seq3_full"       seq_3 "$GT_SEQ3" "$SCRIPT_DIR/main.py"

PATCH=$(make_patch_no_ego)
[ -n "$PATCH" ] && run_and_eval "seq3_no_ego"     seq_3 "$GT_SEQ3" "$PATCH"

PATCH=$(make_patch_with_kf)
[ -n "$PATCH" ] && run_and_eval "seq3_with_kf"    seq_3 "$GT_SEQ3" "$PATCH"

# ══════════════════════════════════════════════════════════
# Final comparison table
# ══════════════════════════════════════════════════════════
echo ""; echo -e "${SEP}"; echo -e "${GREEN}FINAL COMPARISON TABLE${NC}"; echo -e "${SEP}"

python3 - "$OUTPUT_ROOT" << 'PYEOF'
import sys, csv as _csv
from pathlib import Path

root   = Path(sys.argv[1])
order  = ["seq1_full","seq1_no_offline","seq1_kf_predict",
          "seq3_full","seq3_no_ego","seq3_with_kf"]
keys   = ["MOTA_proxy","FP_points","FN_points","IDSW_proxy","fragmentation_proxy"]
G,R,Y,B,NC = "\033[0;32m","\033[0;31m","\033[1;33m","\033[1m","\033[0m"
lw, cw = 22, 13

header = f"  {B}{'Experiment':<{lw}}"
for k in keys: header += f"  {k[:cw]:>{cw}}"
print(header + NC)
print("  " + "─"*(lw+(cw+2)*len(keys)))

prev = None
for exp in order:
    p = root/f"{exp}_score.csv"
    seq = "_".join(exp.split("_")[:2])
    if seq != prev and prev is not None:
        print("  " + "─"*(lw+(cw+2)*len(keys)))
    prev = seq
    if not p.exists():
        print(f"  {exp:<{lw}}  (not run)"); continue
    with open(p) as f:
        r = list(_csv.DictReader(f))[0]
    row = f"  {exp:<{lw}}"
    for k in keys:
        v = float(r.get(k,0))
        if k=="MOTA_proxy":
            col = G if v>=0.95 else (Y if v>=0.7 else R)
            row += f"  {col}{v:>{cw}.4f}{NC}"
        else:
            col = G if v==0 else (Y if v<50 else R)
            row += f"  {col}{int(v):>{cw}}{NC}"
    print(row)
print("")
PYEOF

# cleanup
rm -f /tmp/main_no_offline.py /tmp/main_no_ego.py \
      /tmp/main_kf_predict.py /tmp/main_with_kf.py \
      /tmp/sdc_tracker_track1_kfpred.py /tmp/sdc_tracker_withkf.py

echo -e "${GREEN}All done. Results → $OUTPUT_ROOT${NC}"