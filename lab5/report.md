# Lab 5 — EKF Sensor Fusion (Radar Odometry + GPS)

## 1. 目標

使用 Extended Kalman Filter 融合 **radar odometry**（高頻、會漂移的相對位姿）與 **GPS**（低頻、絕對但有雜訊的 xy），輸出貼合 groundtruth 的軌跡。

## 2. 系統設計

### 狀態
```
x_t = [x, y, yaw]ᵀ
```

### Motion model（predict）
控制量 `u = [dx, dy, dyaw]` 為 radar odom 在前一時刻 **車體座標系** 下的增量。

```
x_{t+1} = x_t + dx·cos(yaw) − dy·sin(yaw)
y_{t+1} = y_t + dx·sin(yaw) + dy·cos(yaw)
yaw_{t+1} = wrap(yaw + dyaw)
```

Jacobian：
```
A = | 1  0  −dx·sin(yaw) − dy·cos(yaw) |
    | 0  1   dx·cos(yaw) − dy·sin(yaw) |
    | 0  0                            1 |
```

### Observation model（update）
GPS 只提供 xy：
```
z = [gps_x, gps_y]ᵀ
H = | 1  0  0 |
    | 0  1  0 |
```

### Kalman 更新
- Predict：`S ← A·S·Aᵀ + R`
- Gain：`K = S·Hᵀ·(H·S·Hᵀ + Q)⁻¹`
- Update：`pose ← pose + K·(z − H·pose)`
- Covariance（Joseph form，保持對稱）：`S ← (I−KH)·S·(I−KH)ᵀ + K·Q·Kᵀ`

## 3. 中間遇到的問題與解決

### 問題 1：第一次 odometry callback 就 crash
`fusion.py` 原本在 `if not self.initial` 判斷前就 unpack `self.last_odom`，而 `self.last_odom` 初始為 `None` → 第一次 callback 直接 TypeError。

**解決**：把 control 計算搬到 `else:` 分支內，保證 `self.last_odom` 已被初始化才會被讀取。

### 問題 2：`AttributeError: predict_EKF`
`fusion.py` 呼叫 `self.EKF.predict_EKF(...)` 但 `EKF.py` 定義的方法名是 `predict`。

**解決**：EKF 類別同時提供 `predict` 與 `predict_EKF`（alias），使兩邊相容。

### 問題 3：觀測矩陣維度不符
`EKF.py` 預設 `C = I(3)`（3×3），但 GPS 量測只是 `[x, y]`（2 維），矩陣乘法會失敗。

**解決**：改 `C` 為 2×3，對應 `Q` 為 2×2。

### 問題 4：Angle wrapping
yaw 在 ±π 交界處會跳變（例如 179° → −179°），造成 `d_yaw` 與 innovation 出現 ~2π 的錯誤修正量。

**解決**：
- predict 後對 yaw 做 `wrap_angle = atan2(sin(θ), cos(θ))`
- 計算 `d_yaw` 亦做 wrap
- update 後再 wrap 一次
- 協方差矩陣改用 Joseph form 避免浮點誤差造成非對稱

### 問題 5：估計軌跡整體偏離 groundtruth 約 5 m
首次跑完 RMSE 很高，從 result 圖看到：起始段貼合、中段起估計軌跡整體平移在 GT 上方約 5 m。

**分析**：原本 `self.EKF.R = data.pose.covariance[:3,:3]`，把 radar odom 訊息內的 covariance 直接當成 process noise。但：
1. 該 covariance 幾乎為 0（radar 節點對自身 over-confident）
2. 其語意是「絕對位姿不確定性」而非「單步 u 的雜訊」

`S` 始終很小 → Kalman gain 給 GPS 的權重接近 0 → GPS 無法把 yaw bias 拉回 → 軌跡整體偏移。

**解決**：改用 **隨每步移動距離縮放的 process noise**：
```python
dist = sqrt(dx_local**2 + dy_local**2)
sigma_xy  = 0.05 + 0.05 * dist              # 位置：5 cm + 5% of step
sigma_yaw = 0.5 * pi / 180                  # 航向：每步 0.5°
self.EKF.R = diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])
```
讓 `S` 隨時間合理成長，GPS 獲得足夠權重，間接修正 yaw drift。調整後估計軌跡完美貼合 GT。

### 問題 6：Headless 環境無法用 rviz
透過 SSH 連線、X forwarding 非常慢。

**解決**：
- `matplotlib.use('Agg')` 改用無 GUI backend，離線存 PNG
- `plot_path()` 結束時印 `RMSE / max error`，不依賴視覺化也能量化評估
- launch 檔將 rviz node 註解掉

## 4. 修改摘要

### `src/fusion/src/EKF.py`
- 新增 `wrap_angle` helper
- `C` 改 2×3（GPS 只觀測 xy），`Q` 改 2×2
- predict / update 加入 yaw wrap
- update 改用 Joseph form 保持 S 對稱
- 新增 `predict_EKF` alias 相容舊呼叫

### `src/fusion/src/fusion.py`
- `matplotlib.use('Agg')` 放在 import pyplot 之前（headless）
- `odometryCallback` 中 control 計算移入 `else` 分支，避免首次 callback unpack `None`
- `d_yaw` 也做 angle wrap
- **process noise R 改為隨移動距離縮放的固定模型**（核心改動）
- `plot_path` 輸出 `samples / RMSE / max error` 並移除 `plt.show()`

### `src/fusion/launch/EKF_fusion.launch`
- 註解掉 rviz node（SSH 無 GUI）


## 5. 使用步驟

### 環境啟動
```bash
# Host 端
cd ~/Projects/ROS1_ws/SDC_lab
docker compose up -d ros
docker exec -it ros bash
```

### 編譯（container 內，一次即可）
```bash
cd /root/catkin_ws
catkin_make
source devel/setup.bash
chmod +x src/fusion/src/fusion.py
```

### 執行（兩個 container shell）

Shell A — EKF 節點
```bash
docker exec -it ros bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
roslaunch fusion EKF_fusion.launch
```

Shell B — 播 rosbag
```bash
docker exec -it ros bash
source /opt/ros/noetic/setup.bash
rosbag play --clock /root/catkin_ws/data/sdc_hw5.bag
```

### 結果取得
bag 播完後在 Shell A 按 `Ctrl+C`，終端印出：
```
[EKF] samples=XXXX  RMSE=X.XXX m  max=X.XXX m
```
PNG 存於 `/root/catkin_ws/result/result1.png`，從 host 拉回：
```bash
docker cp ros:/root/catkin_ws/result/result1.png ~/Projects/ROS1_ws/SDC_lab/lab5/
```

### 監看（可選）
```bash
rostopic hz /pred       # check EKF output 
rostopic hz /gt_odom    # check bag play
```

## 6. 結果

調整 process noise 後，估計軌跡已完美貼合 groundtruth。
