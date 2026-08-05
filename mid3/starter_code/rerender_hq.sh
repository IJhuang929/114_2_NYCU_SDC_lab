#!/bin/bash
# ============================================================
# rerender_hq.sh — 高畫質重跑所有 seq，產生所有影片
#
# 產出：
#   seq1_offline_comparison_hq.mp4
#   seq2_full_hq.mp4
#   seq3_ego_comparison_hq.mp4
#   seq4_full_hq.mp4
#
# Usage: cd starter_code && bash rerender_hq.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUB_ROOT="../public_dev"
TST_ROOT="../test_input"
OUTPUT_ROOT="../outputs/ablation_hq"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

mkdir -p "$OUTPUT_ROOT"

# ── patch helpers ─────────────────────────────────────────
make_hq_main() {
    python3 - << 'PYEOF' >&2
with open("main.py") as f:
    src = f.read()
src = src.replace("dpi=160", "dpi=220")
with open("/tmp/main_hq.py", "w") as f:
    f.write(src)
print("  main_hq: dpi=220 patched")
PYEOF
}

make_hq_no_ego() {
    python3 - << 'PYEOF' >&2
with open("/tmp/main_hq.py") as f:
    src = f.read()
src = src.replace(
    "ego_delta = ego_motion[frame_idx - 1]",
    "ego_delta = None  # ABLATION: ego comp disabled"
)
with open("/tmp/main_hq_no_ego.py", "w") as f:
    f.write(src)
print("  main_hq_no_ego patched")
PYEOF
}

make_hq_no_offline() {
    python3 - << 'PYEOF' >&2
with open("/tmp/main_hq.py") as f:
    src = f.read()
src = src \
    .replace("ENABLE_OFFLINE_STITCHING = True", "ENABLE_OFFLINE_STITCHING = False") \
    .replace("ENABLE_SWAP_CORRECTION = True",   "ENABLE_SWAP_CORRECTION = False")
with open("/tmp/main_hq_no_offline.py", "w") as f:
    f.write(src)
print("  main_hq_no_offline patched")
PYEOF
}

# ── runner: only vis path goes to stdout, everything else to stderr ──
run_seq() {
    local label=$1 seq=$2 data_root=$3 script=$4
    local out_dir="$OUTPUT_ROOT/$label"
    mkdir -p "$out_dir"
    echo -e "${YELLOW}>>> [$label] running ...${NC}" >&2
    PYTHONPATH="$SCRIPT_DIR" python3 "$script" \
        --data-root "$data_root" \
        --seq "$seq" \
        --output-root "$out_dir" \
        --max-age 8 --min-hits 1 --max-distance 5.0 \
        --plot-every 1 >&2
    local vis
    vis=$(find "$out_dir" -type d -name "tracking_vis" | head -1)
    local n; n=$(find "$vis" -name "*.png" 2>/dev/null | wc -l)
    echo -e "${GREEN}  vis → $vis  ($n frames)${NC}" >&2
    # only the path goes to stdout
    echo "$vis"
}

# ── single video ───────────────────────────────────────────
make_single_video() {
    local vis_dir=$1 out_mp4=$2 fps=${3:-10}
    echo -e "${YELLOW}>>> Making single video: $(basename $out_mp4) ...${NC}"
    python3 - "$vis_dir" "$out_mp4" "$fps" << 'PYEOF'
import sys, cv2
from pathlib import Path

vis_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
fps = int(sys.argv[3])
frames = sorted(vis_dir.glob("frame_*.png"))
if not frames:
    print(f"ERROR: no frames in {vis_dir}"); import sys; sys.exit(1)
probe = cv2.imread(str(frames[0]))
h, w = probe.shape[:2]
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i, f in enumerate(frames):
    img = cv2.imread(str(f))
    if img is not None:
        writer.write(img)
    if i % 100 == 0:
        print(f"  {i}/{len(frames)} frames...")
writer.release()
print(f"[DONE] {out_path}  ({len(frames)/fps:.1f}s @ {fps}fps)")
PYEOF
}

# ── comparison video ───────────────────────────────────────
make_comparison_video() {
    local left=$1 right=$2 out=$3 label_left=$4 label_right=$5
    shift 5
    echo -e "${YELLOW}>>> Making comparison video: $(basename $out) ...${NC}"
    python3 make_comparison_video.py \
        --left  "$left" \
        --right "$right" \
        --out   "$out" \
        --fps   10 \
        --label-left  "$label_left" \
        --label-right "$label_right" \
        "$@"
}

# ══════════════════════════════════════════════════════════
echo -e "${GREEN}Patching scripts ...${NC}"
make_hq_main
make_hq_no_ego
make_hq_no_offline

# ── SEQ 1 ─────────────────────────────────────────────────
echo ""; echo "── SEQ 1 ──────────────────────────────────────"
SEQ1_FULL_VIS=$(run_seq  "seq1_full_hq"       seq_1 "$PUB_ROOT" /tmp/main_hq.py)
SEQ1_NOOFF_VIS=$(run_seq "seq1_no_offline_hq" seq_1 "$PUB_ROOT" /tmp/main_hq_no_offline.py)
echo "seq1_full vis: $SEQ1_FULL_VIS"
echo "seq1_nooff vis: $SEQ1_NOOFF_VIS"
make_comparison_video \
    "$SEQ1_FULL_VIS" "$SEQ1_NOOFF_VIS" \
    "$OUTPUT_ROOT/seq1_offline_comparison_hq.mp4" \
    "With offline stitching (IDSW=2)" \
    "Without offline stitching (IDSW=49)" \
    --panel right

# ── SEQ 2 ─────────────────────────────────────────────────
echo ""; echo "── SEQ 2 ──────────────────────────────────────"
SEQ2_FULL_VIS=$(run_seq "seq2_full_hq" seq_2 "$TST_ROOT" /tmp/main_hq.py)
echo "seq2_full vis: $SEQ2_FULL_VIS"
make_single_video "$SEQ2_FULL_VIS" "$OUTPUT_ROOT/seq2_full_hq.mp4" 10

# ── SEQ 3 ─────────────────────────────────────────────────
echo ""; echo "── SEQ 3 ──────────────────────────────────────"
SEQ3_FULL_VIS=$(run_seq  "seq3_full_hq"   seq_3 "$PUB_ROOT" /tmp/main_hq.py)
SEQ3_NOEGO_VIS=$(run_seq "seq3_no_ego_hq" seq_3 "$PUB_ROOT" /tmp/main_hq_no_ego.py)
echo "seq3_full vis: $SEQ3_FULL_VIS"
echo "seq3_noego vis: $SEQ3_NOEGO_VIS"
make_comparison_video \
    "$SEQ3_FULL_VIS" "$SEQ3_NOEGO_VIS" \
    "$OUTPUT_ROOT/seq3_ego_comparison_hq.mp4" \
    "With ego compensation (IDSW=31)" \
    "Without ego compensation (IDSW=65)" \
    --panel right

# ── SEQ 4 ─────────────────────────────────────────────────
echo ""; echo "── SEQ 4 ──────────────────────────────────────"
SEQ4_FULL_VIS=$(run_seq "seq4_full_hq" seq_4 "$TST_ROOT" /tmp/main_hq.py)
echo "seq4_full vis: $SEQ4_FULL_VIS"
make_single_video "$SEQ4_FULL_VIS" "$OUTPUT_ROOT/seq4_full_hq.mp4" 10

# ── Summary ───────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}ALL DONE — Videos:${NC}"
for f in "$OUTPUT_ROOT"/*.mp4; do
    size=$(du -h "$f" | cut -f1)
    echo -e "  ${GREEN}$size${NC}  $f"
done

rm -f /tmp/main_hq.py /tmp/main_hq_no_ego.py /tmp/main_hq_no_offline.py