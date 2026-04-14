# 4D Radar Localization - 實作報告

## 專案概述

本專案使用 **4D 雷達點雲** 搭配 **NDT (Normal Distributions Transform) Scan Matching** 演算法，將每幀雷達掃描對齊到已知的 lidar 地圖，估算車輛位姿 (x, y, yaw)。

整條資料管線：
```
ROS bag (ARS548 radar) → radar_preprocess.py (疊加3幀, xyz PCD) → localization.cpp (NDT matching) → result_*.txt
```

---

## 核心演算法：Coarse-to-Fine NDT Scan Matching

NDT 將目標點雲（地圖）劃分為固定大小的體素格子，每個格子內的點分佈以常態分佈建模。透過最佳化源點雲（雷達）在目標分佈上的機率，迭代求解最佳的剛體變換矩陣。

本專案採用 **兩階段 Coarse-to-Fine** 策略，先用大體素快速收斂，再用小體素精修。NDT 輸入為 **Z 軸壓平後的點雲**（radar 和 submap 都設 z=0），讓配準只在 x, y 平面上進行：

### NDT 參數設定

| 階段       | `resolution` | `maxIterations` | 說明                   |
| ---------- | ------------ | --------------- | ---------------------- |
| Coarse     | 3.0 m        | 15              | 大網格，快速粗對齊     |
| Fine       | 1.0 m        | 20              | 小網格，精修（需 Coarse 收斂才執行） |

| 共用參數                | 值   | 說明               |
| ----------------------- | ---- | ------------------ |
| `transformationEpsilon` | 1e-2 | 收斂判斷門檻       |
| `stepSize`              | 0.1  | 牛頓法搜索步長     |

### Point Type

全部管線統一使用 `pcl::PointXYZ`：
- **Lidar map**：`test.pcd`（預處理後），以 PointXYZ 載入
- **Radar PCD**：`radar_preprocess.py` 輸出 3 幀疊加的 xyz PCD，以 PointXYZ 載入

---

## 優化策略

### 1. 地圖預處理（離線，`map_modified.cpp`）

對原始 lidar 地圖進行 VoxelGrid 降採樣，減少點數以加速後續匹配：

| 方法      | 參數              | 效果             |
| --------- | ----------------- | ---------------- |
| VoxelGrid | leaf size = 0.4 m | 降低點密度       |

輸出：`test.pcd`（PointXYZ 格式）

### 2. 雷達點雲預處理（離線，`radar_preprocess.py`）

從 ROS bag 提取雷達點雲，只取 x, y, z 欄位，並以滑動窗口疊加 3 幀增加點數密度：

| 參數       | 值    | 說明                                   |
| ---------- | ----- | -------------------------------------- |
| 讀取欄位   | x,y,z | 不讀取 RCS/RangeRate                   |
| 疊加幀數   | 3     | deque(maxlen=3) 滑動窗口               |
| 單幀點數   | ~350  | 疊加後約 ~1050 點                      |

輸出：`stack_3_frame_radar_pcd/*.pcd`

### 3. 外參變換（radar_front → base_link）

Radar 點雲在 sensor frame（`radar_front`）下，需先轉換到車體座標系（`base_link`）才能和地圖做 NDT。

外參鏈從 `/tf_static` 取得：`base_link → top_122219002200 (≈Identity) → radar_front`

```
T_base_to_radar ≈ [[1,  0,  0,  2.022],
                    [0,  1,  0,  0.359],
                    [0,  0,  1, -1.15 ],
                    [0,  0,  0,  1    ]]
```

每幀載入 PCD 後立刻套用 `T_radar_to_base = T_base_to_radar⁻¹` 轉換。

### 4. 動態 Submap 裁切

以當前估計位置為中心，使用 CropBox 裁切 **200m x 200m** 區域作為 NDT 匹配目標，避免對整張地圖做匹配。包含空 submap 防護：若裁切結果為空，直接使用 init_guess 並跳過該幀。

### 5. Z 軸壓平（2D NDT）

4D 雷達在高度方向的解析度較差，3D NDT 會被 z 軸的噪音干擾。將 radar 和 submap 的 z 座標全部設為 0，init_guess 的 z 也壓平，讓 NDT 只在 x, y 平面上配準。輸出的 z 值固定為 0。

### 6. Initial Guess 策略

- **第一幀**：hardcoded 初始位姿（4x4 矩陣），z 設為 0
- **後續幀**：直接使用上一幀的定位結果作為 init_guess

### 7. 發散保護

兩個條件任一觸發即拒絕該幀 NDT 結果：

| 條件       | 門檻                      | 原因                           |
| ---------- | ------------------------- | ------------------------------ |
| 位移跳躍   | > 8.0 m                   | NDT 結果與預測偏差過大         |
| NDT 未收斂 | `hasConverged() == false` | 結果不可信                     |

拒絕時直接使用 `init_guess` 作為當前幀位姿。

### 8. 診斷記錄

每幀輸出至 `diag_*.txt`，記錄：
- `frame`：幀編號
- `score`：NDT fitness score
- `jump_distance`：與 init_guess 的位移差
- `diverged`：是否發散（0/1）

結尾附加總幀數與發散幀數統計。

---

## 處理流程

```
啟動
  |
  +-- 載入預處理地圖 (test.pcd, PointXYZ)
  +-- 計算 radar_front → base_link 外參 (T_radar_to_base)
  |
  +-- 逐幀迴圈：
       |
       (1) 讀取雷達 PCD 檔案 (PointXYZ, 已疊加 3 幀)
       |
       (2) 外參變換：radar_front → base_link
       |
       (3) 計算 init_guess
       |   +-- 第一幀：hardcoded 初始位姿
       |   +-- 後續幀：使用上一幀位姿
       |
       (4) CropBox 裁切 submap (200m x 200m)
       |   +-- 空 submap → 跳過，使用 init_guess
       |
       (5) Z 軸壓平 (radar + submap 都設 z=0)
       |
       (6) Coarse-to-Fine NDT Scan Matching
       |   +-- Coarse: resolution=3.0, iter=15
       |   +-- Fine:   resolution=1.0, iter=20 (需 Coarse 收斂)
       |
       (7) 發散保護（jump > 8.0m 或未收斂 → 使用 init_guess）
       |
       (8) Z 固定為 0
       |
       (9) 發布 ROS topic + 寫入 result_*.txt + 寫入 diag_*.txt
```

---

## ROS Topics

| Topic                    | 型別               | 內容               |
| ------------------------ | ------------------ | ------------------ |
| `/4D_radar`              | PointCloud2        | 當前雷達點雲       |
| `/4D_radar_submap_pc`    | PointCloud2        | 裁切後的 lidar submap |
| `/map_pc`                | PointCloud2        | 完整地圖           |
| `/tranformed_radar_pose` | PoseStamped        | 每幀定位 pose      |
| `/localization_path`     | nav_msgs/Path      | 累積軌跡           |

---

## 檔案結構

```
localization/
+-- src/
|   +-- localization.cpp      # 主程式：NDT 定位
|   +-- map_modified.cpp      # 地圖預處理工具
|   +-- visualization.cpp     # 離線結果回放
+-- launch/
|   +-- localization.launch   # 執行定位（支援 use_rviz 參數）
|   +-- map_modified.launch   # 執行地圖預處理
|   +-- visualization.launch  # 離線視覺化（支援 use_rviz 參數）
+-- config/
|   +-- localization.rviz     # 定位用 RViz 設定
|   +-- visualization.rviz    # 回放用 RViz 設定
+-- processing_codes/
|   +-- radar_preprocess.py   # ROS bag → PCD 轉換（疊加 3 幀, xyz only）
|   +-- txt_to_idx_x_y_yaw.py # result.txt → CSV（idx, x, y, yaw）
|   +-- eval_score.py         # 軌跡平滑度評估（無需 GT）
+-- data/
|   +-- Track1/               # 測試軌跡資料
|       +-- Track1.bag        # 原始 ROS bag
|       +-- map.pcd           # 原始 lidar 地圖
|       +-- test.pcd          # 預處理後地圖
|       +-- init_guess.txt    # 初始位姿
|       +-- radar_t_list.txt  # 雷達時間戳列表
|       +-- stack_3_frame_radar_pcd/  # 疊加 3 幀的雷達 PCD (xyz)
+-- result/
    +-- result_<timestamp>.txt # 定位輸出（3x4 變換矩陣，每行 12 個數字）
    +-- diag_<timestamp>.txt   # 診斷記錄（每幀 score, jump_distance, diverged）
```

---

## 完整執行步驟

### Step 0：環境準備

SSH 連入遠端 Docker container，確認 ROS 1 workspace 已設定：
```bash
source /opt/ros/noetic/setup.bash
cd /root/catkin_ws
```

### Step 1：雷達點雲前處理（Python，只需跑一次）

從 ROS bag 提取雷達點雲，疊加 3 幀並只取 xyz 存成 PCD：
```bash
python3 src/localization/processing_codes/radar_preprocess.py
```
- 輸入：`data/Track1/Track1.bag`（topic: `/ars548/radar_front/detections`）
- 輸出：`data/Track1/stack_3_frame_radar_pcd/*.pcd`

### Step 2：地圖預處理（C++，只需跑一次）

對原始 lidar 地圖做 VoxelGrid 降採樣：
```bash
catkin_make
source devel/setup.bash
roslaunch localization map_modified.launch
```
- 輸入：`data/Track1/map.pcd`
- 輸出：`data/Track1/test.pcd`

### Step 3：執行定位

```bash
# 有 GUI 環境（本地或 X11 forwarding）
roslaunch localization localization.launch

# 無 GUI 環境（SSH headless）
roslaunch localization localization.launch use_rviz:=false
```
- 輸出：`result/result_<timestamp>.txt` + `result/diag_<timestamp>.txt`
- 終端會即時顯示每幀 log：`Frame N | Score: x.xx | Jump: x.xx | x: ... y: ... | yaw: ...`

### Step 4：結果檢視與評估

```bash
# 轉成 CSV 格式（idx, x, y, yaw）供繪圖或上傳 Kaggle
python3 src/localization/processing_codes/txt_to_idx_x_y_yaw.py \
    result/result_<timestamp>.txt result/result.csv

# 軌跡平滑度評估（不需要 ground truth）
python3 src/localization/processing_codes/eval_score.py result/result_<timestamp>.txt

# 查看診斷記錄（每幀 NDT score / jump / 是否發散）
cat result/diag_<timestamp>.txt
```

---

## 評分方式

Kaggle 使用 **RMSE of ATE (Absolute Trajectory Error)**：

```
E_n = sqrt(1/n * Σ||Δp_i||²) + k * sqrt(1/n * Σ||Δθ_i||²)
```

- `Δp`：平移誤差（公尺）
- `Δθ`：旋轉誤差（度）
- `k = 0.2`（旋轉權重）

無 ground truth 時可透過以下指標間接判斷定位品質：

| 觀察指標            | 對應評分項 | 怎麼看                                      |
| ------------------- | ---------- | -------------------------------------------- |
| NDT Fitness Score   | Δp         | `diag_*.txt` 中 score 越小越好               |
| Jump Distance       | Δp         | `diag_*.txt` 中 jump 持續偏大代表不穩定      |
| 發散次數            | Δp + Δθ   | `diag_*.txt` 最後一行的 diverged_frames      |
| 幀間位移標準差      | Δp         | `eval_score.py` 的 displacement std 越小越好 |
| 幀間 yaw 變化       | Δθ         | `eval_score.py` 的 yaw change 突變代表問題   |

---

## 可調參數與改善方向

詳見 `parameter.md`，包含所有可調參數的說明與建議實驗順序。

### 目前可考慮的改善方向

1. **重新啟用 CV Model**（帶速度 clamp）：高速段 init_guess 偏差太大，NDT 容易掉入局部最小值
2. **Submap VoxelGrid 降採樣**：submap 點太密，加速 NDT
3. **重新啟用強度距離補償**：`compensated_intensity = 4*log(R) + log(Pr)`，過濾雜訊點（需搭配 PointXYZI + radar_pcd_with_intensity）
4. **Frame Stacking in code**：在 localization.cpp 內做多幀疊加（取代 preprocessing 的 3 幀疊加），可動態調整 buffer 大小
5. **RadiusOutlierRemoval**：去除孤立雜點
