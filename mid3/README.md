# 4D Radar Multi-Object Tracking

Student IDs: 314605011 / 314611105

## Overview

Two trackers are implemented:

- **Competition 1 (seq_1 / seq_2)** — `sdc_tracker_track1.py`
  Kalman filter + Hungarian algorithm with Re-ID (lost track pool). Offline post-processing: tracklet stitching and ID-swap correction after all frames are processed.

- **Competition 2 (seq_3 / seq_4)** — `sdc_tracker_track2.py`
  Same Kalman filter core with ego-motion compensation using `ego_global_pos.txt`. Per-frame incremental CSV output.

Both trackers are invoked through the unified `starter_code/main.py` entry point, which selects the correct tracker based on the sequence name.

## Files

```
starter_code/
├── main.py                  Unified entry point (auto-selects tracker by seq)
├── sdc_tracker_track1.py    Tracker for seq_1 / seq_2
├── sdc_tracker_track2.py    Tracker for seq_3 / seq_4 (ego-motion aware)
├── sdc_tracking_utils.py    Data loading and CSV writing utilities
├── evaluate_tracking.py     Local MOTA evaluator (public dev only)
├── check_result_format.py   Validates result.csv format
├── run.sh                   Runs and evaluates seq_1 and seq_3
├── run_ablation.sh          Runs ablation variants
└── requirements.txt         Python dependencies
competition_1_result.csv     Best submission for Competition 1 (seq_2)
competition_2_result.csv     Best submission for Competition 2 (seq_4)
```

## Directory Layout

Before running, place the data folders at the same level as `starter_code/`:

```
314605011_314611105_tracking/
├── starter_code/
├── public_dev/
│   ├── seq_1/
│   └── seq_3/
└── test_input/
    ├── seq_2/
    └── seq_4/
```

## Install

```bash
cd starter_code
pip install -r requirements.txt
```

## Run

### Competition 1 — seq_1 (public dev, with evaluation)

```bash
cd starter_code
python main.py --data-root ../public_dev --seq seq_1 --output-root ../outputs
python evaluate_tracking.py \
  --gt ../public_dev/seq_1/gt_answer_seq1.csv \
  --pred ../outputs/seq_1_*/result.csv \
  --out ../outputs/seq_1_eval.txt
```

### Competition 1 — seq_2 (generate submission)

```bash
cd starter_code
python main.py --data-root ../test_input --seq seq_2 --output-root ../outputs
# submission CSV: ../outputs/seq_2_*/result.csv
```

### Competition 2 — seq_3 (public dev, with evaluation)

```bash
cd starter_code
python main.py --data-root ../public_dev --seq seq_3 --output-root ../outputs
python evaluate_tracking.py \
  --gt ../public_dev/seq_3/gt_answer_seq3.csv \
  --pred ../outputs/seq_3_*/result.csv \
  --out ../outputs/seq_3_eval.txt
```

### Competition 2 — seq_4 (generate submission)

```bash
cd starter_code
python main.py --data-root ../test_input --seq seq_4 --output-root ../outputs
# submission CSV: ../outputs/seq_4_*/result.csv
```

### Run both sequences at once

```bash
cd starter_code
bash run.sh
```

## Ablation Study

To reproduce the ablation experiments (compares variants such as Re-ID, KF prediction, tentative tracks, ego compensation):

```bash
cd starter_code
bash run_ablation.sh
```

Results are saved to `../outputs/ablation/`. Each variant's MOTA and error counts are printed to the terminal and written as CSV files.

## Optional arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--max-age N` | Frames a track survives without a match | 5 (seq1/2), 8 (seq3/4) |
| `--min-hits N` | Hits required to confirm a track | 3 (seq1/2), 1 (seq3/4) |
| `--max-distance F` | Association distance gate (metres) | 5.0 (seq1/2), 4.0 (seq3/4) |
| `--plot-every N` | Save visualization every N frames | 0 (disabled) |
