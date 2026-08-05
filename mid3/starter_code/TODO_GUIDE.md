# TODO Guide

This guide explains the minimum implementation steps needed to turn the starter framework into a working tracker.

The intended workflow is:

```text
radar points + cluster mask
        ↓
cluster-level detection extraction
        ↓
track prediction / association / update
        ↓
cluster_id -> track_id mapping
        ↓
result.csv
```

## Minimum working solution

To obtain a meaningful `result.csv`, complete these TODO blocks in order.

---

## TODO-1: Build cluster-level detections

File:

```text
main.py
```

Function:

```text
calculate_cluster_centroids()
```

Goal:

Convert moving radar points into one detection per moving cluster.

Suggested implementation:

1. Find all unique cluster IDs in `moving_mask`.
2. Ignore invalid labels if any appear.
3. For each cluster ID:
   - select all radar points belonging to that cluster
   - compute mean `[x, y, z]`
   - compute mean `range_rate`
   - store the original points for optional analysis

Expected outputs:

```text
cluster_centroids[cluster_id] = np.array([x, y, z])
cluster_velocities[cluster_id] = mean_range_rate
cluster_points_dict[cluster_id] = points
```

---

## TODO-4: Build detection array

File:

```text
main.py
```

Function:

```text
build_detections()
```

Goal:

Convert dictionaries from TODO-1 into a NumPy array that the tracker can process.

Suggested detection format:

```text
[x, y, z, cluster_id, range_rate]
```

Expected shape:

```text
num_detections x 5
```

If you change this detection format, also update `sdc_tracker.py` and `generate_cluster_track_dict()` accordingly.

---

## TODO-5: Call tracker update

File:

```text
main.py
```

Location:

```text
run_sequence()
```

Goal:

Pass current-frame detections into the tracker.

Suggested implementation:

```python
confirmed_tracks = tracker.update(detections)
```

Students may also replace the provided `Tracker` with their own tracker.

---

## TODO-6: Implement data association / gating

File:

```text
sdc_tracker.py
```

Function:

```text
assign_detections_to_tracks()
```

Goal:

Match current detections to existing tracks.

Basic approach:

1. Compute pairwise distances between tracks and detections.
2. Use a matching strategy.
3. Reject matches with distance larger than `max_distance`.
4. Return:
   - matched track-detection pairs
   - unmatched detections
   - unmatched tracks

Possible methods:

```text
nearest-neighbor matching
Hungarian assignment
Mahalanobis gating
Kalman Filter prediction
range_rate-aware cost
RCS-aware cost
ego-motion compensated cost
```

For Hungarian assignment, `scipy.optimize.linear_sum_assignment` is recommended.

---

## TODO-2: Map confirmed tracks back to clusters

File:

```text
main.py
```

Function:

```text
generate_cluster_track_dict()
```

Goal:

Convert confirmed tracks into the dictionary required by `save_tracking_mask_to_csv()`.

Required format:

```python
cluster_track_dict = {
    cluster_id: track_id,
}
```

This mapping controls which radar points receive confirmed tracking IDs in `result.csv`.

---

## TODO-3: Optional visualization

File:

```text
main.py
```

Function:

```text
visualize_tracking()
```

This TODO is optional. It does not affect Kaggle submission if left empty.

Useful visualization elements:

```text
static radar points
moving clusters
cluster centroids
track IDs
track history / trajectory tails
```

---

# Suggested improvement directions

## Competition 1

Focus on stable cluster-to-track association.

Possible improvements:

```text
better association cost
motion model
Kalman Filter
track lifecycle tuning
range_rate usage
gating threshold tuning
```

## Competition 2

Focus on moving-ego tracking.

Possible improvements:

```text
ego motion compensation using ego_global_pos.txt
ego-aware association cost
motion model under ego movement
range_rate / RCS / cluster size features
robust track lifecycle management
```

# Common mistakes

1. Producing a CSV with correct format but no confirmed tracking IDs.
2. Returning track IDs that are not mapped back to cluster IDs.
3. Using different detection formats in `main.py` and `sdc_tracker.py`.
4. Forgetting to check result format before Kaggle submission.
5. Evaluating on hidden test GT. Only public development GT should be used locally.
