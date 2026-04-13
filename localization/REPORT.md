# 4D Radar Localization - 實作報告

## 專案概述

本專案使用 **4D 雷達點雲** 搭配 **NDT (Normal Distributions Transform) Scan Matching** 演算法，將每幀雷達掃描對齊到已知地圖，估算車輛位姿 (x, y, yaw)。

## 核心演算法：NDT Scan Matching

NDT 將目標點雲（地圖）劃分為固定大小的體素格子，每個格子內的點分佈以常態分佈建模。透過最佳化源點雲（雷達）在目標分佈上的機率，迭代求解最佳的剛體變換矩陣。

### NDT 參數設定

| 參數                    | 值    | 說明                               |
| ----------------------- | ----- | ---------------------------------- |
| `resolution`            | 1.0 m | NDT 體素格子大小，越小越精確但越慢 |
| `transformationEpsilon` | 1e-3  | 收斂判斷門檻                       |
| `stepSize`              | 0.1   | 牛頓法搜索步長                     |
| `maxIterations`         | 100   | 最大迭代次數                       |

---

## 優化策略

### 1. 地圖預處理（離線，`map_modified.cpp`）

對原始地圖進行三階段過濾，減少點數以加速後續匹配：

| 階段     | 方法                        | 參數              | 效果                   |
| -------- | --------------------------- | ----------------- | ---------------------- |
| z 軸過濾 | PassThrough                 | z ∈ [-20, 30]    | 去除地面以下與天空雜點 |
| 離群去除 | Statistical Outlier Removal | k=20, stddev=1.5  | 去除孤立噪音點         |
| 降採樣   | VoxelGrid                   | leaf size = 0.3 m | 降低點密度             |

**結果：6565 萬點 → 986 萬點**

### 2. 動態 Submap 裁切

以當前估計位置為中心，使用 CropBox 裁切 **200m x 200m** 區域作為匹配目標，避免對整張地圖做匹配。

裁切後再以 VoxelGrid（leaf size = 0.5m）進行二次降採樣，需小於 NDT resolution（1.0m）以確保每個體素有足夠點數建模。

### 3. 雷達點雲預處理

#### 3a. 距離過濾

保留距離 **5m ~ 60m** 範圍的雷達點：

| 過濾範圍   | 原因                                     |
| ---------- | ---------------------------------------- |
| < 5m 去除  | 近處多為移動物體（前車、行人），干擾匹配 |
| > 60m 去除 | 遠處雷達解析度低、噪音大，位置不準確     |

#### 3b. 雷達點雲降採樣

使用 VoxelGrid（leaf size = 0.5m）對雷達點雲降採樣，使不同幀的點密度一致，NDT 匹配品質更穩定。

### 4. 2D 匹配（Z 軸壓平）

4D 雷達在高度方向的解析度較差，直接做 3D 匹配會導致 z 方向漂移。

- **複製**點雲副本，將 z 全部設為 0
- init_guess 的 z 同步壓平
- NDT 僅在 x, y, yaw 上匹配
- **保留原始點雲**給 RViz 顯示，不影響視覺化

### 5. Initial Guess 策略

- **第一幀**：從 `init_guess.txt` 讀取初始位姿
- **後續幀**：直接使用上一幀的定位結果作為 init_guess

### 6. 異常幀過濾（Outlier Rejection）

五個條件任一觸發即拒絕該幀結果：

| 條件          | 門檻                      | 原因                                 |
| ------------- | ------------------------- | ------------------------------------ |
| 位移跳躍      | > 3.0 m                   | 正常幀約 0.2~0.5m，3m 已屬異常       |
| yaw 突變      | > 0.3 rad（~17.2 度）     | 車輛不會一幀內轉這麼大角度           |
| NDT 未收斂    | `hasConverged() == false` | 結果不可信                           |
| Fitness Score | > 5.0                     | 匹配品質太差，結果不可靠             |
| 倒退偵測      | dot < -0.1 且 jump > 0.3  | 位移方向與車頭方向相反，物理上不合理 |

#### 倒退偵測

計算 NDT 位移向量與車頭朝向的內積（dot product）。若內積為負且位移量顯著（> 0.3m），判定為倒退，拒絕該幀。

#### Reject 處理

- **正常 reject**：使用上一幀位姿作為當前幀結果，跳過 EMA 平滑
- **連續 reject ≥ 3 次**：若 NDT 已收斂且非倒退，強制接受但做幅度限制（clamping）：

| 限制項   | 最大值                     |
| -------- | -------------------------- |
| 位移     | 2.0 m（等比例縮放 dx, dy） |
| yaw 變化 | 0.1 rad（~5.7 度）         |

避免因連續拒絕導致車輛永遠卡在同一位置。

### 7. 直線偵測與自適應 EMA 平滑

這是針對長直線場景的核心優化。在直線段上，道路兩側特徵稀疏且重複，NDT 容易在橫向（lateral）和 yaw 方向產生抖動，但縱向（longitudinal）仍然可靠。

#### 7a. 直線偵測

維護最近 **20 幀** 的 yaw 歷史，使用 unwrapped 角度計算標準差（避免 wrap-around 問題）。若標準差 < **0.5 度**，判定為直線行駛。

#### 7b. 位移分解

將 NDT 結果與 init_guess 的位移差分解為兩個分量：

- **縱向（longitudinal）**：沿車頭方向的分量（`dx*cos(yaw) + dy*sin(yaw)`）
- **橫向（lateral）**：垂直車頭方向的分量（`-dx*sin(yaw) + dy*cos(yaw)`）

#### 7c. 暖機期

前 **20 幀**（`STRAIGHT_HISTORY_SIZE`）歷史不足，無法準確判斷直線/彎道，統一使用保守 alpha（lateral = 0.15, yaw = 0.15），避免開頭左右晃動。

#### 7d. 自適應 Alpha 決策

根據「直線 vs 彎道」和「NDT 品質」兩個維度，動態調整 EMA 平滑的 alpha 值：

| 狀態                | NDT 品質判定              | alpha_lon | alpha_lat | alpha_yaw |
| ------------------- | ------------------------- | --------- | --------- | --------- |
| **直線 + NDT 差**   | score > 1.5 or iter > 30  | 0.4       | 0.02      | 0.02      |
| **直線 + NDT 普通** | 其他                      | 0.7       | 0.15      | 0.15      |
| **直線 + NDT 好**   | score < 0.5 and iter < 15 | 0.7       | 0.4       | 0.4       |
| **彎道 + NDT 差**   | score > 1.5 or iter > 30  | 0.7       | 0.3       | 0.3       |
| **彎道 + NDT 普通** | 其他                      | 0.7       | 0.7       | 0.7       |
| **彎道 + NDT 好**   | score < 0.5 and iter < 15 | 0.7       | 0.9       | 0.9       |

#### 7e. 橫向硬上限

不管 alpha 算出多少，單幀橫向位移最多 **0.5m**，防止極端情況下的橫向跳躍。

平滑後將縱向和橫向分量重新組合回 x, y 座標，yaw 則獨立平滑。

### 8. Pitch / Roll 約束

車輛在平面上行駛，只有 x, y, yaw 三個自由度。旋轉矩陣強制重建為僅含 yaw 的形式，pitch 和 roll 歸零，z 值鎖定為初始值。

---

## 處理流程

```
啟動
  |
  +-- 載入預處理地圖 (test.pcd)
  +-- 設定 NDT 參數（一次性）
  |
  +-- 逐幀迴圈：
       |
       (1) 讀取雷達 PCD 檔案
       |
       (2) 計算 init_guess
       |   +-- 第一幀：讀取 init_guess.txt
       |   +-- 後續幀：使用上一幀位姿
       |
       (3) CropBox 裁切 submap (200m x 200m)
       |
       (4) VoxelGrid 降採樣 submap (leaf = 0.5m)
       |
       (5) 距離過濾雷達點 (5m ~ 60m)
       |
       (6) VoxelGrid 降採樣雷達點 (leaf = 0.5m)
       |
       (7) 複製點雲 + Z 軸壓平
       |
       (8) NDT Scan Matching
       |
       (9) 異常幀過濾（跳躍 / yaw / 收斂 / score / 倒退）
       |   +-- 正常 reject → 使用上一幀位姿
       |   +-- 連續 reject >= 3 次 → 強制接受（clamped）
       |
       (10) 直線偵測（yaw 歷史標準差 < 0.5 度）
       |
       (11) 自適應 EMA 平滑（僅 accepted 幀）
       |   +-- 暖機期（前 20 幀）→ 保守 alpha
       |   +-- 分解位移為縱向 / 橫向
       |   +-- 根據直線狀態 + NDT 品質選擇 alpha
       |   +-- 橫向硬上限 0.5m
       |   +-- 重組位移 + 獨立平滑 yaw
       |
       (12) Pitch / Roll 歸零，Z 鎖定
       |
       (13) 更新上一幀位姿
       |
       (14) 發布 ROS topic + 寫入 result_<timestamp>.txt
```

---

## 檔案結構

```
localization/
+-- src/
|   +-- localization.cpp      # 主程式：NDT 定位
|   +-- map_modified.cpp      # 地圖預處理工具
|   +-- visualization.cpp     # 離線結果回放
+-- launch/
|   +-- localization.launch   # 執行定位 + RViz
|   +-- map_modified.launch   # 執行地圖預處理
|   +-- visualization.launch  # 離線視覺化
+-- config/
|   +-- localization.rviz     # 定位用 RViz 設定
|   +-- visualization.rviz    # 回放用 RViz 設定
+-- processing_codes/
|   +-- radar_preprocess.py   # ROS bag -> PCD 轉換
|   +-- txt_to_idx_x_y_yaw.py # 結果轉 CSV
|   +-- eval_score.py         # 軌跡品質評估
+-- data/
|   +-- Track1/               # 測試軌跡 1
|   +-- Track2/               # 測試軌跡 2
+-- result/
    +-- result_<timestamp>.txt # 定位輸出（3x4 變換矩陣，帶時間戳）
```

---

## 執行方式

```bash

# 1. 地圖預處理（只需執行一次）
roslaunch localization map_modified.launch

# 2. 編譯
catkin_make
source devel/setup.bash

# 3. 執行定位
roslaunch localization localization.launch

# 4. 評估結果
python3 processing_codes/eval_score.py result/result_<timestamp>.txt

# 5. 結果轉 CSV
python3 processing_codes/txt_to_idx_x_y_yaw.py result/result_<timestamp>.txt result/result.csv
```


## 4/14
1. claude 確認code邏輯功能正確
2. 跑新的 radar point cloud
3. 實測看看這個版本
