# SDC Lab - Radar-LiDAR NDT Localization Report

## 1. System Overview

本系統使用 4D 毫米波雷達點雲與 LiDAR 地圖進行 NDT (Normal Distributions Transform) scan matching 定位。整體 pipeline 分為三個階段：

```
LiDAR Map (map.pcd)          Raw Radar PCD (1800 frames)
       |                              |
  map_modified.cpp             radar_preprocessor.py
       |                              |
 preprocessed_map.pcd     preprocessed_radar_pcd/*.pcd
                                + ego_velocity.csv
                \                /
                 localization.cpp
                       |
                result_*.txt --> txt_to_idx_x_y_yaw.py --> .csv (Kaggle)
```

所有參數統一由 `config/params.yaml` 管理，實驗時只需修改此檔案。

---

## 2. Stage 1: LiDAR Map Preprocessing (`map_modified.cpp`)

**功能**: 將原始 LiDAR 地圖點雲進行清理與降採樣，產生乾淨的定位用地圖。

**Pipeline**:
1. **載入原始地圖** - 讀取 `lidar_map_pcd_path` 指定的 PCD 檔
2. **高度濾波 (PassThrough)** - 移除 z 軸範圍外的點 (`lidar_z_min` ~ `lidar_z_max`)
3. **RANSAC 地面移除** - 使用 `SACMODEL_PLANE` 擬合地面平面，移除地面點 (`ransac_distance_threshold`)
4. **Voxel Grid 降採樣** - 以 `lidar_voxel_size` 為 leaf size 進行體素降採樣
5. **Statistical Outlier Removal** - 移除離群點 (`sor_mean_k`, `sor_std_dev`)
6. **儲存 & 發布** - 儲存為 `preprocessed_map_save_path`，並發布至 `/lidar_map` 和 `/lidar_map_for_ndt` topics (latched)

**相關參數** (`params.yaml`):
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `lidar_map_pcd_path` | Track1/map.pcd | 原始地圖路徑 |
| `preprocessed_map_save_path` | Track1/preprocessed_map.pcd | 輸出路徑 |
| `lidar_z_min` | -1.0 | 高度下限 (m) |
| `lidar_z_max` | 4.0 | 高度上限 (m) |
| `ransac_distance_threshold` | 0.2 | RANSAC 平面距離閾值 (m) |
| `lidar_voxel_size` | 0.3 | 降採樣 leaf size (m) |
| `sor_mean_k` | 30 | SOR 鄰域點數 |
| `sor_std_dev` | 1.0 | SOR 標準差倍率 |

---

## 3. Stage 2: Radar Preprocessing (`radar_preprocessor.py`)

**功能**: 對原始 4D 雷達點雲進行濾波、ego velocity 估測與多幀累積。

**Pipeline**:
1. **單幀濾波** - RCS 閾值、高度範圍、距離範圍篩選
2. **Ego Velocity RANSAC 估測** - 利用 Doppler (RangeRate) 觀測模型 `v_r = -D @ v_ego` 進行 RANSAC，其中 D = [cos(az)cos(el), sin(az)cos(el), sin(el)]，200 次迭代，3 點取樣，再以所有 inlier 重新擬合
3. **多幀累積** - 滑動視窗累積 `accumulate_frames` 幀，以 ego velocity 進行運動補償（平移），最後 voxel downsample

**輸出**:
- `preprocessed_radar_pcd/{timestamp_nsec}.pcd` - 每幀累積後的點雲 (fields: x y z intensity)
- `preprocessed_radar_pcd/ego_velocity.csv` - 每幀 ego velocity (timestamp_nsec, vx, vy, vz)

**相關參數** (`params.yaml`):
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `radar_pcd_folder` | Track1/raw_radar_pcd/ | 原始雷達 PCD 目錄 |
| `rcs_min_threshold` | -10.0 | RCS 最小閾值 (dBsm) |
| `radar_z_min` | -1.5 | 雷達高度下限 (m) |
| `radar_z_max` | 3.0 | 雷達高度上限 (m) |
| `range_min` | 1.5 | 最小距離 (m) |
| `range_max` | 80.0 | 最大距離 (m) |
| `doppler_inlier_threshold` | 0.15 | Doppler RANSAC inlier 閾值 (m/s) |
| `accumulate_frames` | 10 | 累積幀數 |
| `accumulate_voxel_size` | 0.3 | 累積後降採樣 leaf size (m) |

---

## 4. Stage 3: NDT Localization (`localization.cpp`)

**功能**: 讀取 preprocessed 資料，執行 two-stage NDT scan matching 定位。

### 4.1 Pipeline (每幀)

1. **Init Guess 決定**
   - 第一幀：使用 `params.yaml` 的 `init_guess` (row-major 3x4 matrix)
   - 後續幀：上一幀結果 + ego velocity prediction (`R_map * v_ego * dt`)

2. **Submap 擷取** (帶快取)
   - 以 init_guess 位置為中心，`submap_size` 為邊長，CropBox 裁切地圖
   - 只在移動距離 > `submap_rebuild_dist` 時重建，否則複用快取

3. **Coarse NDT** (可跳過)
   - Resolution: `ndt_coarse_voxel` (1.5m)，最大 20 次迭代
   - 當上一幀 fitness score < `coarse_skip_threshold` 時跳過，直接使用 init_guess

4. **Fine NDT + RCS Weighting**
   - RCS 加權：計算 RCS z-score -> sigmoid -> 映射至 [0.5, 2.0]，weight > 1.0 的點複製一份
   - Resolution: `ndt_fine_voxel` (0.5m)，最大 `ndt_max_iterations` 次迭代

5. **Degeneracy Detection**
   - 將對齊後的雷達點與 submap 做 KdTree nearest neighbor search (每 3 個點取樣 1 個)
   - 計算殘差協方差矩陣 H 的特徵值比 (lambda_max / lambda_min)
   - 若比值 > `degenerate_ratio_threshold`，在退化方向上將位移縮減 80%

6. **Pose Validation & Smoothing**
   - Fitness score > `fitness_fail_threshold` 或未收斂 -> 保持上一幀 pose
   - 跳躍距離 > `max_jump_distance` -> 拒絕此幀
   - EMA 平滑：`pose = alpha * ndt_result + (1-alpha) * prev_pose`

### 4.2 加速優化

| 優化策略 | 說明 |
|---------|------|
| Submap 快取 | 移動 > 30m 才重建 submap 和 KdTree |
| Submap 縮小 | 200m -> 100m |
| Coarse NDT 跳過 | 上一幀 score 好時直接跳過 coarse 階段 |
| NDT 迭代縮減 | Coarse 30->20, Fine 50->30 |
| Degeneracy 降採樣 | 每 3 個點取 1 個 + 複用快取 KdTree |

### 4.3 ROS Topics

| Topic | 類型 | 說明 |
|-------|------|------|
| `/4D_radar` | PointCloud2 | 當前幀雷達點雲 |
| `/4D_radar_submap_pc` | PointCloud2 | 當前 submap |
| `/tranformed_radar_pose` | PoseStamped | 定位 pose |
| `/localization_path` | Path | 累積軌跡 |
| `/localization/fitness_score` | Float32 | NDT fitness score |
| `/localization/is_degenerate` | Bool | 是否退化 |
| `/localization/status` | String | OK / FAILED / JUMP_DETECTED / DEGENERATE |

### 4.4 輸出檔案

- `result/result_{timestamp}.txt` - 每行為 3x4 row-major 變換矩陣 (12 個數字)
- `result/diag_{timestamp}.txt` - 診斷 CSV (frame, score, x, y, yaw, radar_pts, submap_pts, converged, status, eigen_ratio)

### 4.5 相關參數 (`params.yaml`)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `radar_t_list` | Track1/radar_t_list.txt | timestamp list 路徑 |
| `preprocessed_radar_pcd_folder` | Track1/preprocessed_radar_pcd | preprocessed radar PCD 目錄 |
| `preprocessed_map_path` | Track1/preprocessed_map.pcd | preprocessed map 路徑 |
| `ego_velocity_csv` | .../ego_velocity.csv | ego velocity CSV 路徑 |
| `result_save_dir` | result/ | 結果輸出目錄 |
| `ndt_coarse_voxel` | 1.5 | Coarse NDT resolution (m) |
| `ndt_fine_voxel` | 0.5 | Fine NDT resolution (m) |
| `ndt_max_iterations` | 30 | Fine NDT 最大迭代數 |
| `ndt_epsilon` | 0.001 | NDT 收斂閾值 |
| `fitness_fail_threshold` | 50.0 | Score 失敗閾值 |
| `max_jump_distance` | 1.5 | 跳躍偵測距離 (m) |
| `degenerate_ratio_threshold` | 100.0 | 退化偵測特徵值比 |
| `pose_smooth_alpha` | 0.7 | EMA 平滑係數 |
| `submap_size` | 100.0 | Submap 邊長 (m) |
| `submap_rebuild_dist` | 30.0 | Submap 重建距離閾值 (m) |
| `coarse_skip_threshold` | 10.0 | 跳過 coarse 的 score 閾值 |
| `init_guess` | [12 floats] | 初始位姿 (row-major 3x4) |

---

## 5. 完整執行步驟

所有操作在 Docker container `ros` 內執行。

### Step 0: 進入 container 並編譯

```bash
docker exec -it ros bash
source /opt/ros/noetic/setup.bash
cd /root/catkin_ws
catkin_make
source devel/setup.bash
```

### Step 1: 修改 `params.yaml`

根據要跑的 Track 修改 `config/params.yaml` 中的路徑與 `init_guess`：

**Track 1**:
```yaml
# IO paths (map_modified)
lidar_map_pcd_path: "/root/catkin_ws/src/localization/data/Track1/map.pcd"
preprocessed_map_save_path: "/root/catkin_ws/src/localization/data/Track1/preprocessed_map.pcd"

# IO paths (radar_preprocessor)
radar_pcd_folder: "/root/catkin_ws/src/localization/data/Track1/raw_radar_pcd/"

# IO paths (localization)
radar_t_list: "/root/catkin_ws/src/localization/data/Track1/radar_t_list.txt"
preprocessed_radar_pcd_folder: "/root/catkin_ws/src/localization/data/Track1/preprocessed_radar_pcd"
preprocessed_map_path: "/root/catkin_ws/src/localization/data/Track1/preprocessed_map.pcd"
ego_velocity_csv: "/root/catkin_ws/src/localization/data/Track1/preprocessed_radar_pcd/ego_velocity.csv"

# Track1 init_guess
init_guess: [-0.7739014871732518, -0.632632984298013, 0.029188959029395573, -25.31523704528809,
             0.632582897383501, -0.7743982245509299, -0.012094120483231403, -4.389283657073975,
             0.030255017582620166, 0.009104778446398086, 0.9995007438318981, -7.73]
```

**Track 2**:
```yaml
# IO paths (map_modified)
lidar_map_pcd_path: "/root/catkin_ws/src/localization/data/Track2/map.pcd"
preprocessed_map_save_path: "/root/catkin_ws/src/localization/data/Track2/preprocessed_map.pcd"

# IO paths (radar_preprocessor)
radar_pcd_folder: "/root/catkin_ws/src/localization/data/Track2/raw_radar_pcd/"

# IO paths (localization)
radar_t_list: "/root/catkin_ws/src/localization/data/Track2/radar_t_list.txt"
preprocessed_radar_pcd_folder: "/root/catkin_ws/src/localization/data/Track2/preprocessed_radar_pcd"
preprocessed_map_path: "/root/catkin_ws/src/localization/data/Track2/preprocessed_map.pcd"
ego_velocity_csv: "/root/catkin_ws/src/localization/data/Track2/preprocessed_radar_pcd/ego_velocity.csv"

# Track2 init_guess
init_guess: [-0.9478046334338953, -0.3188028892577441, 0.005576257002181274, 499.4871520996094,
              0.3186884878163892, -0.9466153206969891, 0.04854979253344068, -163.97898864746094,
             -0.010199243822116278, 0.047792807227150104, 0.9988051977251651, 7.679019927978516]
```

### Step 2: LiDAR Map Preprocessing

```bash
roslaunch localization map_modified.launch
```

輸出: `preprocessed_map.pcd` (根據 `preprocessed_map_save_path` 路徑)

完成後 Ctrl+C 結束。

### Step 3: Radar Preprocessing

```bash
cd /root/catkin_ws/src/localization/processing_codes
python3 radar_preprocessor.py ../config/params.yaml
```

輸出:
- `preprocessed_radar_pcd/{timestamp}.pcd` (每幀一個檔案)
- `preprocessed_radar_pcd/ego_velocity.csv`

### Step 4: 執行 Localization

```bash
roslaunch localization localization.launch
```

如果不需要 rviz 視覺化 (例如 SSH 環境):
```bash
roslaunch localization localization.launch use_rviz:=false
```

輸出:
- `result/result_{timestamp}.txt` - 定位結果
- `result/diag_{timestamp}.txt` - 診斷資訊

### Step 5: 轉換結果為 Kaggle 提交格式

```bash
cd /root/catkin_ws/src/localization/processing_codes
python3 txt_to_idx_x_y_yaw.py <result.txt路徑> <output.csv路徑>
```

範例:
```bash
python3 txt_to_idx_x_y_yaw.py \
    /root/catkin_ws/src/localization/result/result_20260415_143022.txt \
    /root/catkin_ws/src/localization/result/track1_submission.csv
```

輸出 CSV 格式:
```
idx,x,y,yaw
0,-25.32,-4.39,2.458
1,-25.45,-4.52,2.461
...
```

將此 `.csv` 檔上傳至 Kaggle 即可。

---

## 6. 檔案結構

```
localization/
├── config/
│   ├── params.yaml              # 所有參數統一管理
│   └── localization.rviz        # rviz 設定
├── launch/
│   ├── map_modified.launch      # LiDAR map preprocessing
│   └── localization.launch      # NDT localization + rviz
├── src/
│   ├── map_modified.cpp         # LiDAR map preprocessor node
│   └── localization.cpp         # NDT localization node
├── processing_codes/
│   ├── radar_preprocessor.py    # Radar preprocessing CLI
│   └── txt_to_idx_x_y_yaw.py   # Result txt -> Kaggle csv
├── data/
│   └── Track{1,2}/
│       ├── map.pcd              # 原始 LiDAR 地圖
│       ├── preprocessed_map.pcd # 處理後地圖 (Step 2 產生)
│       ├── raw_radar_pcd/       # 原始雷達 PCD
│       ├── preprocessed_radar_pcd/  # 處理後雷達 PCD (Step 3 產生)
│       │   ├── {timestamp}.pcd
│       │   └── ego_velocity.csv
│       └── radar_t_list.txt     # timestamp 列表
└── result/
    ├── result_*.txt             # 定位結果 (Step 4 產生)
    └── diag_*.txt               # 診斷資訊
```
