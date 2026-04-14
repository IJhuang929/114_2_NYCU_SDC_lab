# Tunable Parameter Reference

> Source: `localization/src/localization.cpp`
> Evaluation metric: Kaggle ATE = RMSE(delta_p) + 0.2 * RMSE(delta_theta)

---

## 1. NDT Registration

| Parameter | Line | Current | Effect (increase) | Effect (decrease) | Priority |
|---|---|---|---|---|---|
| Coarse Resolution | L324 | `3.0` | Coarser grid, wider convergence basin but lower precision | Finer grid, local minima risk | Medium |
| Fine Resolution | L331 | `1.0` | Lower precision, faster | Higher precision, slower, local minima risk | **High** |
| Coarse MaxIterations | L325 | `25` | More iterations, slower, better convergence | Faster but may stop early | Low |
| Fine MaxIterations | L332 | `20` | Same as above | Same as above | Low |
| TransformationEpsilon | L316 | `1e-3` | Stops earlier (tolerant), faster | Stricter convergence, more precise but slower | Medium |
| StepSize | L317 | `0.1` | Larger search step, faster but may overshoot | Smaller step, precise but slow | **High** |

### Notes
- `Fine Resolution` is the most sensitive NDT param. With sparse radar (~250-350 pts), 1.0 may be too fine. Try 1.5 or 2.0.
- `StepSize` 0.1 is conservative. Try 0.05 (more precise) or 0.2 (faster).
- Coarse-to-Fine strategy: coarse stage handles large displacement, fine stage refines. If coarse doesn't converge, fine stage is skipped.

---

## 2. Radar Point Cloud Filtering

| Parameter | Line | Current | Effect (increase) | Effect (decrease) | Priority |
|---|---|---|---|---|---|
| Distance min | L251 | `5.0` m | Removes more near-range noise (body reflection) | Keeps close points, may add noise | Low |
| Distance max | L251 | `60.0` m | Keeps far points (radar less accurate at range) | Only uses near high-quality points | Medium |
| INTENSITY_THRESHOLD | L247 | `3.0` | Stricter, fewer but cleaner points | More lenient, more points | **High** |
| Radar VoxelGrid | L268 | OFF (was 0.5m) | Larger leaf = fewer points | Smaller leaf = more points | Low (radar already sparse) |
| RadiusOutlierRemoval | L288 | OFF (was r=3.0, minN=2) | Larger radius = keep more; larger minN = stricter | Opposite | Medium |

### Intensity Distance Compensation
```
compensated_intensity = 4 * log(dist) + log(Pr)
```
- Based on radar equation: received power decays with R^4
- `Pr` = raw intensity (RCS mapped), clamped to 1e-6
- Compensated value is used for both filtering and as the point's intensity field
- INTENSITY_THRESHOLD filters on this compensated value, NOT raw RCS

### Suggested experiments
- INTENSITY_THRESHOLD: test 2.0 / 3.0 / 4.0 / 5.0, observe `after_filter` point count in log
- Distance max: try 40.0 / 50.0 / 60.0, far radar points have poor localization accuracy
- RadiusOutlierRemoval: re-enable after NDT is stable, try r=2.0 minN=2

---

## 3. Submap Extraction

| Parameter | Line | Current | Effect (increase) | Effect (decrease) | Priority |
|---|---|---|---|---|---|
| square_size | L208 | `100.0` m | Larger submap, slower, tolerates larger init_guess error | Faster, but may miss map if init_guess drifts | Medium |
| Submap VoxelGrid leaf | L297 | `0.5` m | Fewer submap points, faster NDT, coarser matching | More points, slower, finer matching | **High** |

### Notes
- Submap size should cover radar range (60m) plus margin for init_guess error. 100m = 50m half-size is sufficient.
- Submap VoxelGrid: test 0.3 / 0.5 / 1.0. Observe `Submap: raw=xxx, downsampled=xxx` in log.

---

## 4. Frame Stacking (Radar Buffer)

| Parameter | Line | Current | Effect (increase) | Effect (decrease) | Priority |
|---|---|---|---|---|---|
| BUFFER_SIZE | L71 | `10` | More history frames stacked, denser point cloud, but accumulated pose error | Fewer frames, sparser but fresher | **High** |

### Notes
- Double-edged sword: more frames = denser cloud for NDT, but old frames carry pose drift
- **Turning scenarios**: large heading change between frames makes old buffer frames harmful
- History frames are transformed to current frame using: `T_history_to_current = init_guess.inverse() * frame.pose`
- If score degrades over time, try reducing buffer (5 or 8)
- Suggested experiments: 5 / 8 / 10 / 15

---

## 5. Motion Model / Initial Guess

| Parameter | Line | Current | Description | Priority |
|---|---|---|---|---|
| CV Model | L198-201 | **OFF** (uses prev_pose) | init_guess = prev_pose, assumes stationary | **High** |
| Divergence jump threshold | L344 | `3.0` m | NDT result > 3m from init_guess = diverged, fallback | Medium |

### CV Model Analysis
- Currently disabled: `init_guess = prev_pose` (zero velocity assumption)
- Problem at high speed: 30 km/h ~ 8 m/s, frame interval ~0.5s => 4m displacement per frame. Using prev_pose as init_guess is 4m off, NDT may fall into local minimum.
- CV Model formula: `init_guess = prev_pose * (prev_prev_pose.inverse() * prev_pose)`
- Risk: if one frame's NDT result is wrong, delta amplifies the error
- **Recommendation**: re-enable with velocity clamp (e.g., max 5 m/frame) to limit error amplification

### Divergence Protection
- `jump_distance > 3.0` may be too small at high speed (50 km/h => ~7m/frame)
- Check diag.txt for cases where jump > 3 but NDT actually converged correctly
- Consider adaptive threshold or speed-dependent threshold

---

## 6. Z-axis Flattening

| Parameter | Line | Current | Description | Priority |
|---|---|---|---|---|
| Z-flatten | L304-310 | **ON** | Both radar and submap set z=0, NDT matches in x,y only | Already enabled |
| Z restore | L352 | `init_guess z` | After NDT, z is restored from init_guess | - |

### Notes
- Radar has poor z-accuracy compared to lidar. Flattening forces 2D matching, which is more robust.
- Z value is preserved from init_guess (propagated from init_guess.txt).
- This is unlikely to need tuning. Keep it ON.

---

## Suggested Experiment Order

Run baseline first (no intensity filter, no submap VoxelGrid), then A/B test one parameter at a time:

| Order | Parameter | Values to test | What to watch |
|---|---|---|---|
| 1 | INTENSITY_THRESHOLD | 2.0 / 3.0 / 4.0 / 5.0 | after_filter count, score trend |
| 2 | Submap VoxelGrid leaf | 0.3 / 0.5 / 1.0 | speed vs. score trade-off |
| 3 | Re-enable CV Model | with velocity clamp | score at high-speed segments |
| 4 | BUFFER_SIZE | 5 / 8 / 10 | score trend over time, turning segments |
| 5 | Fine Resolution | 1.0 / 1.5 / 2.0 | score, convergence rate |
| 6 | Distance max | 40 / 50 / 60 | after_filter count, score |
| 7 | RadiusOutlierRemoval | r=2.0 minN=2 | noise reduction vs. point loss |
