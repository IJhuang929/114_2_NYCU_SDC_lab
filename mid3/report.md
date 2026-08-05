# Tracking 改進分析報告

## 一、問題發現

對比兩個版本的追蹤器，發現 MOTA 分數有明顯差距。分析後找到以下關鍵差異：

### 現有版本（`starter_code/sdc_tracker.py`）的問題

當一個 confirmed track 因連續幾幀沒有 detection 匹配（超過 `max_age`），會被**直接刪除**。

若同一個物體在下一幀再次出現，tracker 會把它視為全新物體，分配新的 `track_id`。
這會導致：

- **IDSW（ID Switch）增加**：同一輛車被切成多段 tracklet，每次重新出現就換一個 ID
- **MOTA 下降**：MOTA = 1 − (FP + FN + IDSW) / GT，IDSW 直接壓低分數

常見觸發場景：
- 物體短暫被遮蔽（被路牌、其他車輛擋住 1~3 幀）
- Radar 點雲在特定角度反射弱，某幀 detection 消失
- 兩車交會交叉時，cluster 暫時合併再分開

---

## 二、參考版本的做法

對比 `sdc_tracker_yi.py`，該版本實作了 **Re-ID（重識別）** 機制：

- Track 消失後不立刻刪除，而是移入 **Lost Track Pool**
- 後續幀的未配對 detection，會先嘗試與 lost pool 中的 track 配對
- 配對成功 → **復活原 track，沿用舊 track_id**，不產生 IDSW

缺點：該版本的 lost track 使用**靜態位置**（最後一幀的座標），若物體仍在移動，lost track 的位置會與實際出現位置有偏差，導致配對失敗或誤配。

---

## 三、改進方案：Re-ID with KF Prediction

在現有 Kalman Filter tracker 的基礎上加入 Re-ID，並利用 KF 的速度估計讓 lost track **持續外推位置**，解決靜態位置造成的配對誤差。

### 流程

```
每幀 update():
  1. 對所有 active track 做 KF predict
  2. 正常 Hungarian matching（active tracks vs detections）
  3. [新增] 對 unmatched detections 嘗試 Re-ID：
       - Lost track 的 KF 繼續 predict（位置跟著物體預測移動）
       - 與 unmatched detections 做 Hungarian matching
       - 距離 < max_distance * 1.5 且 range_rate 差 < 4.0 m/s → 復活，沿用舊 track_id
  4. 剩下仍未配對的 detections → 開新 track
  5. 超過 max_age 的 active track → 移入 lost pool（而非直接刪除）
  6. Lost track lost_age 超過上限 → 真正刪除
```

### 與 Yi 版本的差異

| 比較項目 | Yi 版本 | 改進後版本 |
|---|---|---|
| Lost track 位置 | 靜止（最後一幀） | **KF 繼續外推**（隨速度移動） |
| Re-ID 距離門檻 | `max_distance × 1.0` | `max_distance × 1.5`（因位置已補償，可稍寬） |
| Range Rate 門檻 | 3.0 m/s | 4.0 m/s |
| 基礎追蹤器 | 無 Kalman Filter | **有 Kalman Filter**（位置估計更平滑） |

### 預期效果

- IDSW 降低 → MOTA 提升
- Lost track 的 KF 外推讓 Re-ID 配對更準確，誤配率低於靜態版本
- 對短暫遮蔽（1~5 幀）的恢復能力明顯提升

---

## 四、實作摘要

修改 `sdc_tracker.py`：

1. `__init__`：加入 `self.lost_tracks = []`、`self.max_lost_age = 15`
2. Age-out 邏輯：confirmed track 超過 `max_age` 時移入 `lost_tracks`，保留 KF 狀態
3. 每幀對 `lost_tracks` 呼叫 `kf.predict()`，讓位置持續外推
4. 新增 `_reid()` method：Hungarian matching between lost tracks and unmatched detections
5. 復活成功的 track 沿用原 `track_id` 重新加入 `self.tracks`

---

## 五、`main_yi.py` 的 Offline 後處理分析

`sdc_tracker_yi.py` + `main_yi.py` 在 seq_1 達到 **MOTA = 0.9962**，高於目前版本。
兩者的 tracker 本身反而更簡單（無 Kalman Filter），關鍵在於 `main_yi.py` 在跑完全部幀後加了兩道後處理。

### 5.1 Offline Tracklet Stitching — 解決「同一物體被切成多段」

**問題根源：** Online tracker 只能看「當下和過去」。物體短暫消失超過 `max_age` 幀後，tracker 刪掉原 track；物體再出現時被當成新物體，得到新 ID，造成 IDSW。

```
真實物體:  ████████████████████████████████
Online ID: ──── A ────  （消失）  ──── B ────
後處理後:  ──── A ────────────────── A ────
```

**做法：** 全部幀跑完後，統計每個 tracklet 的結束位置、結束時間、最後 range rate。對所有 (A, B) 配對計算：

```
score = dist(A_last, B_first) + rr_weight × |rr_A - rr_B| + gap_weight × gap_frames
```

score 夠低（位置近、RR 相近、時間差小）→ 把 B 的所有幀 ID 改成 A。

**為什麼要 offline：** 需要知道 B 之後「沒有再分岔」才能安全合併；online 時還不知道未來。

---

### 5.2 Offline ID Swap Correction — 解決「兩物體交叉時 ID 互換」

**問題根源：** 兩物體靠近交叉時，Hungarian matching 可能在某一幀接錯，之後持續錯下去。

```
物體 1: ────────────╲────────────
物體 2: ────────────╱────────────
                 (交叉點)
Online:  1→1→1→[swap]→2→2→2
         2→2→2→[swap]→1→1→1
```

**做法：** 掃描所有相鄰幀 (t, t+1)，對每對共同存在的 track (A, B)：

```
keep_cost = dist(A_t → A_t+1) + dist(B_t → B_t+1)
swap_cost = dist(A_t → B_t+1) + dist(B_t → A_t+1)
```

若 `swap_cost + margin < keep_cost` 且 gain 夠大 → 從第 t+1 幀開始，把後面所有幀的 A、B ID 全部對調。

**為什麼要 offline：** 需要先確認「對調之後後面是否維持一致」，online 時只有局部資訊，無法判斷。

---

### 5.3 為什麼這兩步對 seq_1 特別有效

seq_1 幾乎沒有真正移動的物體（rr std=0.33，近乎靜止）。靜止物體的雷達點雲容易在某幀完全消失，觸發 track 斷裂；又因為靜止物體聚集，交叉時容易 ID 互換。Stitching 和 Swap Correction 恰好針對這兩種場景。

---

# 目前 based line
| # | MOTA | base reid| tune |  tune + offline switch |
|---|---|---|---|---|
| seq1 | 0.99015  | 0.9908 | 0.9908 | 0.9931 |
| seq2 | 0.967065 | 0.96856 | 0.96856| 0.9745 |
| seq3 | 0.980801 | 0.9832 | 0.985 | 0.986 |
| seq4 | 0.996588 | 0.99317 | 0.9969673 |0.99658|

 python3 main.py   --data-root ../test_input/   --seq seq_4   --output-root ../outputs --max-age 8   --min-hits 1   --max-distance 5.0

python3 main_track1.py   --data-root ../test_input/   --seq seq_2   --output-root ../outputs --max-age 8   --min-hits 1   --max-distance 5.0      


# range rate 分佈
seq_1: rr mean=-0.04, std=0.33, max=1.00
seq_2: rr mean=-0.06, std=0.50, max=17.84
seq_3: rr mean=-3.65, std=2.79, max=45.98
seq_4: rr mean=-6.46, std=4.05, max=8.04
seq_1 幾乎沒有真正移動的物體（rr max=1.0，std=0.33），radar 看到的 range rate 全是感測器雜訊。
seq_2 有真正的移動物體（rr max=17.84，std=0.50），代表有車輛在高速靠近或遠離。
seq3, seq4 兩者都有真實 rr 訊號，但尺度不同

