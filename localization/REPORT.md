# 4D Radar Localization - 實作報告

## 專案概述

本專案使用 **4D 雷達點雲** 搭配 **NDT (Normal Distributions Transform) Scan Matching** 演算法，將每幀雷達掃描對齊到已知地圖，估算車輛位姿 (x, y, yaw)。

---

## 核心演算法：NDT Scan Matching

NDT 將目標點雲（地圖）劃分為固定大小的體素格子，每個格子內的點分佈以常態分佈建模。透過最佳化源點雲（雷達）在目標分佈上的機率，迭代求解最佳的剛體變換矩陣。

### NDT 參數設定

| 參數 | 值 | 說明 |
|---|---|---|
| `resolution` | 1.0 m | NDT 體素格子大小，越小越精確但越慢 |
| `transformationEpsilon` | 1e-3 | 收斂判斷門檻 |
| `stepSize` | 0.1 | 牛頓法搜索步長 |
| `maxIterations` | 100 | 最大迭代次數 |

---

## 優化策略

### 1. 地圖預處理（離線，`map_modified.cpp`）

對原始地圖進行三階段過濾，減少點數以加速後續匹配：

| 階段 | 方法 | 參數 | 效果 |
|---|---|---|---|
| z 軸過濾 | PassThrough | z ∈ [-20, 30] | 去除地面以下與天空雜點 |
| 離群去除 | Statistical Outlier Removal | k=20, stddev=1.5 | 去除孤立噪音點 |
| 降採樣 | VoxelGrid | leaf size = 0.3 m | 降低點密度 |

**結果：6565 萬點 → 986 萬點**

### 2. 動態 Submap 裁切

以當前估計位置為中心，使用 CropBox 裁切 **100m × 100m** 區域作為匹配目標，避免對整張地圖做匹配。

裁切後再以 VoxelGrid（leaf size = 1.0m）進行二次降採樣。

### 3. 雷達點雲距離過濾

保留距離 **5m ~ 60m** 範圍的雷達點：

| 過濾範圍 | 原因 |
|---|---|
| < 5m 去除 | 近處多為移動物體（前車、行人），干擾匹配 |
| > 60m 去除 | 遠處雷達解析度低、噪音大，位置不準確 |

### 4. 2D 匹配（Z 軸壓平）

4D 雷達在高度方向的解析度較差，直接做 3D 匹配會導致 z 方向漂移。

- **複製**點雲副本，將 z 全部設為 0
- init_guess 的 z 同步壓平
- NDT 僅在 x, y, yaw 上匹配
- **保留原始點雲**給 RViz 顯示，不影響視覺化

### 5. 異常幀過濾（Outlier Rejection）

四個條件任一觸發即拒絕該幀結果，沿用上一幀位姿：

| 條件 | 門檻 | 原因 |
|---|---|---|
| 位移跳躍 | > 2.0 m | 正常幀約 0.2~0.5m，2m 已屬異常 |
| yaw 突變 | > 0.15 rad（~8.6°） | 車輛不會一幀內轉這麼大角度 |
| NDT 未收斂 | `hasConverged() == false` | 結果不可信 |
| Fitness Score | > 3.0 | 匹配品質太差，結果不可靠 |

### 6. Pitch / Roll 約束

車輛在平面上行駛，只有 x, y, yaw 三個自由度。NDT 結果僅保留 yaw 旋轉，pitch 和 roll 歸零，z 值鎖定為初始值。

---

## 曾嘗試但移除的方法

### 恆速運動模型（Constant Velocity Model）

利用前兩幀的位姿差預測當前幀位置作為 NDT 初始猜測：

```
delta_T = prev_prev⁻¹ × prev
init_guess = prev × delta_T
```

**移除原因：** 雷達幀率約 20Hz，每幀位移僅 ~0.3m，上一幀位姿已足夠接近真值。恆速模型在某幀 NDT 錯誤時會往錯誤方向外推，導致連鎖發散，反而降低穩定性。改為直接使用上一幀位姿作為 init_guess。

---

## 處理流程

```
啟動
  │
  ├─ 載入預處理地圖 (test.pcd)
  ├─ 設定 NDT 參數（一次性）
  │
  └─ 逐幀迴圈：
       │
       ① 讀取雷達 PCD 檔案
       │
       ② 計算 init_guess
       │   ├─ 第一幀：讀取 init_guess.txt
       │   └─ 後續幀：使用上一幀位姿
       │
       ③ CropBox 裁切 submap (100m × 100m)
       │
       ④ VoxelGrid 降採樣 submap (leaf = 1.0m)
       │
       ⑤ 距離過濾雷達點 (5m ~ 60m)
       │
       ⑥ 複製點雲 + Z 軸壓平
       │
       ⑦ NDT Scan Matching
       │
       ⑧ 異常幀過濾（跳躍 / yaw / 收斂 / score）
       │
       ⑨ Pitch / Roll 歸零，Z 鎖定
       │
       ⑩ 發布 ROS topic → 寫入 result_<timestamp>.txt
```

---

## 檔案結構

```
localization/
├── src/
│   ├── localization.cpp      # 主程式：NDT 定位
│   ├── map_modified.cpp      # 地圖預處理工具
│   └── visualization.cpp     # 離線結果回放
├── launch/
│   ├── localization.launch   # 執行定位 + RViz
│   ├── map_modified.launch   # 執行地圖預處理
│   └── visualization.launch  # 離線視覺化
├── config/
│   ├── localization.rviz     # 定位用 RViz 設定
│   └── visualization.rviz    # 回放用 RViz 設定
├── processing_codes/
│   ├── radar_preprocess.py   # ROS bag → PCD 轉換
│   ├── txt_to_idx_x_y_yaw.py # 結果轉 CSV
│   └── eval_score.py         # 軌跡品質評估
├── data/
│   ├── Track1/               # 測試軌跡 1
│   └── Track2/               # 測試軌跡 2
└── result/
    └── result_<timestamp>.txt # 定位輸出（3×4 變換矩陣，帶時間戳）
```

---

## 執行方式

```bash
# 1. 地圖預處理（只需執行一次）
roslaunch localization map_modified.launch

# 2. 執行定位
roslaunch localization localization.launch

# 3. 評估結果
python3 processing_codes/eval_score.py result/result_<timestamp>.txt

# 4. 結果轉 CSV
python3 processing_codes/txt_to_idx_x_y_yaw.py result/result_<timestamp>.txt result/result.csv
```
