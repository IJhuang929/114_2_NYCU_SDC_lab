# 4D Radar Localization - 實作報告

## 專案概述

本專案使用 **4D 雷達點雲** 搭配 **NDT (Normal Distributions Transform) Scan Matching** 演算法，將每幀雷達掃描對齊到已知的 lidar 地圖，估算車輛位姿 (x, y, yaw)。

整條資料管線：
```
ROS bag (ARS548 radar) → radar_preprocess.py (疊加3幀, xyz PCD) → localization.cpp (NDT matching) → result_*.txt
```

---

## 核心演算法：Single-Pass NDT Scan Matching

NDT 將目標點雲（地圖）劃分為固定大小的體素格子，每個格子內的點分佈以常態分佈建模。透過最佳化源點雲（雷達）在目標分佈上的機率，迭代求解最佳的剛體變換矩陣。

### NDT 參數設定

| 參數                    | 值   | 說明               |
| ----------------------- | ---- | ------------------ |
| `resolution`            | 2.0 m | NDT 體素格子大小   |
| `maxIterations`         | 50   | 最大迭代次數       |
| `transformationEpsilon` | 1e-3 | 收斂判斷門檻       |
| `stepSize`              | 0.1  | 牛頓法搜索步長     |

### Point Type

全部管線統一使用 `pcl::PointXYZ`：
- **Lidar map**：`test.pcd`（預處理後），以 PointXYZ 載入
- **Radar PCD**：3 幀疊加的 PCD（含 x y z RangeRate RCS 欄位），以 PointXYZ 載入（只取前 3 欄）

---

## 優化策略

### 1. 地圖預處理（離線，`map_modified.cpp`）

對原始 lidar 地圖進行 VoxelGrid 降採樣，減少點數以加速後續匹配：

| 方法      | 參數              | 效果             |
| --------- | ----------------- | ---------------- |
| VoxelGrid | leaf size = 0.5 m | 降低點密度       |

輸出：`test.pcd`（PointXYZI 格式，以 PointXYZ 載入）

### 2. 雷達點雲預處理（離線，`radar_preprocess.py`）

從 ROS bag 提取雷達點雲，只取 x, y, z 欄位，並以滑動窗口疊加 3 幀增加點數密度：

| 參數       | 值    | 說明                                   |
| ---------- | ----- | -------------------------------------- |
| 讀取欄位   | x,y,z | 不讀取 RCS/RangeRate                   |
| 疊加幀數   | 3     | deque(maxlen=3) 滑動窗口               |
| 單幀點數   | ~350  | 疊加後約 ~1050 點                      |

輸出：`stack_3_frame_radar_pcd/*.pcd`

注意：預處理的 3 幀疊加是 naive concatenation（無 pose compensation），車在高速移動時點雲會模糊。

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

### 4. 動態 Submap 裁切 + 降採樣

以當前估計位置為中心，使用 CropBox 裁切 **200m x 200m** 區域作為 NDT 匹配目標，避免對整張地圖做匹配。裁切後再以 VoxelGrid（leaf size = 0.5m）降採樣，大幅減少 submap 點數加速 NDT。

包含空 submap 防護：若裁切結果為空，直接使用 init_guess 並跳過該幀。

### 5. Initial Guess 策略：CV Model（恆速模型）

| 情境         | init_guess 來源                        |
| ------------ | -------------------------------------- |
| 第一幀       | hardcoded 初始位姿（從 `init_guess.txt` 取得） |
| 第二幀       | 直接使用第一幀的定位結果               |
| 第三幀起     | CV Model：用前兩幀的 x, y 位移差外推  |

CV Model 以加法外推 x, y，z 和旋轉矩陣保持上一幀的值不動，避免 z 軸飄移。

```
dx = transform_matrix.x - prev_pose.x
dy = transform_matrix.y - prev_pose.y
init_guess = transform_matrix
init_guess.x += dx
init_guess.y += dy
```

### 6. 發散保護

三個條件任一觸發即拒絕該幀 NDT 結果，改用 init_guess：

| 條件           | 門檻                      | 原因                           |
| -------------- | ------------------------- | ------------------------------ |
| 空 submap      | submap 點數 = 0           | 無法匹配                       |
| 往回走         | 內積 < 0 且位移 > 1.0 m   | NDT 結果方向與行進方向相反     |
| 位移跳躍       | > 10.0 m                  | NDT 結果與 init_guess 偏差過大 |
| NDT 未收斂     | `hasConverged() == false` | 結果不可信                     |

拒絕時使用 CV Model 預測的 init_guess，讓車繼續往前走。

### 7. 診斷記錄

每幀輸出至 `diag_*.txt`，記錄：
- `frame`：幀編號
- `score`：NDT fitness score
- `x, y, yaw`：定位結果
- `radar_pts`：雷達點數
- `submap_pts`：降採樣後的 submap 點數
- `converged`：NDT 是否收斂（0/1）
- `rejected`：是否被拒絕（0/1）

結尾附加總幀數統計。

---

## 處理流程

```
啟動
  |
  +-- 載入地圖 (test.pcd / map.pcd, PointXYZ)
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
       |   +-- 第二幀：使用上一幀結果
       |   +-- 第三幀起：CV Model (x, y 外推)
       |
       (4) CropBox 裁切 submap (200m x 200m)
       |
       (5) Submap VoxelGrid 降採樣 (leaf size = 0.5m)
       |
       (6) 空 submap 檢查 → 空則跳過 NDT
       |
       (7) NDT Scan Matching (resolution=2.0, iter=50)
       |
       (8) 發散保護
       |   +-- 往回走 (內積<0 且 dist>1m) → rejected
       |   +-- 跳躍 (dist>10m) → rejected
       |   +-- 未收斂 → rejected
       |   +-- rejected 時使用 init_guess
       |
       (9) 發布 ROS topic + 寫入 result_*.txt + 寫入 diag_*.txt
```

---

## ROS Topics

| Topic                    | 型別               | 內容               |
| ------------------------ | ------------------ | ------------------ |
| `/4D_radar`              | PointCloud2        | 當前雷達點雲       |
| `/4D_radar_submap_pc`    | PointCloud2        | 裁切降採樣後的 submap |
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
|   +-- Track1/               # 測試軌跡 1
|   +-- Track2/               # 測試軌跡 2
|       +-- *.bag             # 原始 ROS bag
|       +-- map.pcd           # 原始 lidar 地圖
|       +-- test.pcd          # 預處理後地圖（VoxelGrid 降採樣）
|       +-- init_guess.txt    # 初始位姿（4x4 矩陣）
|       +-- radar_t_list.txt  # 雷達時間戳列表
|       +-- stack_3_frame_radar_pcd/  # 疊加 3 幀的雷達 PCD
+-- result/
    +-- result_<timestamp>.txt # 定位輸出（3x4 變換矩陣，每行 12 個數字）
    +-- diag_<timestamp>.txt   # 診斷記錄
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
- 輸入：`data/Track*/Track*.bag`（topic: `/ars548/radar_front/detections`）
- 輸出：`data/Track*/stack_3_frame_radar_pcd/*.pcd`

### Step 2：地圖預處理（C++，只需跑一次）

對原始 lidar 地圖做 VoxelGrid 降採樣（leaf size = 0.5m）：
```bash
catkin_make
source devel/setup.bash
roslaunch localization map_modified.launch
```
- 輸入：`data/Track*/map.pcd`
- 輸出：`data/Track*/test.pcd`

注意：需修改 `map_modified.cpp` 中的路徑切換 Track。

### Step 3：執行定位

```bash
# 有 GUI 環境（本地或 X11 forwarding）
roslaunch localization localization.launch

# 無 GUI 環境（SSH headless）
roslaunch localization localization.launch use_rviz:=false
```
- 輸出：`result/result_<timestamp>.txt` + `result/diag_<timestamp>.txt`
- 終端即時顯示：`Frame N | Score: x.xx | x: ... y: ... z: ... | yaw: ... | rejected: 0/1`

注意：需修改 `localization.cpp` 中的路徑和 init_guess 切換 Track。

### Step 4：結果檢視與評估

```bash
# 轉成 CSV 格式（idx, x, y, yaw）供繪圖或上傳 Kaggle
python3 src/localization/processing_codes/txt_to_idx_x_y_yaw.py \
    result/result_<timestamp>.txt result/result.csv

# 軌跡平滑度評估（不需要 ground truth）
python3 src/localization/processing_codes/eval_score.py result/result_<timestamp>.txt

# 查看診斷記錄
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

---

## 可調參數與改善方向

詳見 `parameter.md`，包含所有可調參數的說明與建議實驗順序。

### 目前可考慮的改善方向

1. **Coarse-to-Fine NDT**：先用大 resolution 粗對齊，再用小 resolution 精修
2. **Z 軸壓平（2D NDT）**：將 radar 和 submap 的 z 設為 0，減少 z 軸噪音干擾
3. **強度距離補償**：`compensated_intensity = 4*log(R) + log(Pr)`，過濾雜訊點（需搭配 PointXYZI）
4. **Frame Stacking with pose compensation**：在 localization.cpp 內做多幀疊加並補償 pose，避免 naive stacking 的模糊問題
5. **RadiusOutlierRemoval**：去除孤立雜點
