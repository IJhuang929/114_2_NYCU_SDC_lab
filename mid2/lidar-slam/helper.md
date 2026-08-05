```bash
# Run Track1
TARGET=Track1 python3 main.py  
```
```bash
# 關閉 loop closure（只做 odometry）
ENABLE\_LC=0 TARGET=Track1 python3 main.py
```
```bash
# 調整 LC 參數
TARGET=Track2 LC\_SCORE=0.04 LC\_DIST=6.0 python3 main.py
```
### sensor frame and world frame 轉換注意
在 odometry 和 LC 的 ICP 呼叫前，都先把 world-frame submap 轉回 sensor-local frame

# 目前的版本
"Track3": {
        "ICP_VOXEL":        0.40,
        "ICP_MAX_DIST":     4.0,
        "ICP_MAX_ITER":     80,
        "MAX_DELTA_TRANS":  8.0,
        "ICP_REJECT_SCORE": 2.0,        # Track3 needs looser
        "KEYFRAME_DIST":    1.5,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    7,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        True,
        "LC_DIST":          8.0,
        "LC_SCORE":         0.35,       # fitness: 65%+ inlier ratio
        "LC_MAX_CAND":      15,
        "LC_MIN_SEP":       80,
        "LC_MAX_CORRECTION":  3.0,
        "LC_MAX_YAW_DEG":   35.0,
        "LC_ICP_MAX_DIST":  8.0,
        "LC_ICP_MAX_ITER":  100,
        "LC_ICP_VOXEL":     0.50,
        "LC_FITNESS_DIST":  1.0,        # tight inlier distance
    },
[FINAL STATS]
  Target:                     Track3
  Total frames:               1738
  Keyframes:                  320
  Loop closures applied:      32
  ICP success:                1687/1737 (97.1%)
  ICP reject (score):         50
  ICP reject (delta_trans):   0
  ICP fail (other):           0
  LC ICP total:               0
  LC ICP success:             0/0 (0.0%)
  LC ICP fail:                0
  Start->End distance:        3.100 m
  Cumulative path length:     433.376 m
  Max single-frame movement:  0.604 m

// main_v2.py round
"Track3": {
        "ICP_VOXEL":        0.40,
        "ICP_MAX_DIST":     4.0,
        "ICP_MAX_ITER":     80,
        "MAX_DELTA_TRANS":  8.0,
        "ICP_REJECT_SCORE": 2.0,        
        "KEYFRAME_DIST":    1.5,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    7,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        True,
        "LC_DIST":          8.0,
        "LC_SCORE":         0.35,       
        "LC_MAX_CAND":      15,
        "LC_MIN_SEP":       80,
        "LC_MAX_CORRECTION":  3.0,
        "LC_MAX_YAW_DEG":   35.0,
        "LC_ICP_MAX_DIST":  8.0,
        "LC_ICP_MAX_ITER":  100,
        "LC_ICP_VOXEL":     0.50,
        "LC_FITNESS_DIST":  1.0,       
        "LC_PGO_BATCH":     1,           
    },
Target:                     Track3
  Total frames:               1738
  Keyframes:                  325
  Loop closures applied:      39
  ICP success:                1675/1737 (96.4%)
  ICP reject (score):         62
  ICP reject (delta_trans):   0
  ICP fail (other):           0
  LC ICP total:               0
  LC ICP success:             0/0 (0.0%)
  LC ICP fail:                0
  Start->End distance:        0.790 m
  Cumulative path length:     439.226 m
  Max single-frame movement:  2.048 m

// LC_score=0.35
"Track3": {
        "ICP_VOXEL":        0.40,
        "ICP_MAX_DIST":     4.0,
        "ICP_MAX_ITER":     80,
        "MAX_DELTA_TRANS":  8.0,
        "ICP_REJECT_SCORE": 2.0,        
        "KEYFRAME_DIST":    1.5,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    7,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        True,
        "LC_DIST":          8.0,
        "LC_SCORE":         0.35,       
        "LC_MAX_CAND":      15,
        "LC_MIN_SEP":       80,
        "LC_MAX_CORRECTION":  3.0,
        "LC_MAX_YAW_DEG":   35.0,
        "LC_ICP_MAX_DIST":  8.0,
        "LC_ICP_MAX_ITER":  100,
        "LC_ICP_VOXEL":     0.50,
        "LC_FITNESS_DIST":  1.0,
        "LC_PGO_BATCH":     3,          # accumulate 3 loops before PGO
        "LC_PGO_COOLDOWN":  15,         # kf to skip LC after each PGO
        "LC_HIST_DEDUP":    30,         # min kf gap before same hist_idx reused               
    },
        
[FINAL STATS]
  Target:                     Track3
  Total frames:               1738
  Keyframes:                  318
  Loop closures applied:      9
  ICP success:                1686/1737 (97.1%)
  ICP reject (score):         51
  ICP reject (delta_trans):   0
  ICP fail (other):           0
  LC ICP total:               0
  LC ICP success:             0/0 (0.0%)
  LC ICP fail:                0
  Start->End distance:        0.644 m
  Cumulative path length:     431.556 m
  Max single-frame movement:  0.838 m

── Proxy metrics (lower → better ATE on Kaggle) ──────
  Drift rate  (end/path):     0.0015  ✓ good
  Smoothness  (RMS jerk):     0.0481 m/f²  ✓
  Lap gap     (halfway):    40.675 m  ✗
  Lap gap     (end    ):    0.644 m  ✓

// LC_score=0.35
"Track3": {
        "ICP_VOXEL":        0.40,
        "ICP_MAX_DIST":     4.0,
        "ICP_MAX_ITER":     80,
        "MAX_DELTA_TRANS":  8.0,
        "ICP_REJECT_SCORE": 2.0,        # [FIX #2] Track3 needs looser
        "KEYFRAME_DIST":    1.5,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    7,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        True,
        "LC_DIST":          8.0,
        "LC_SCORE":         0.35,       # fitness: 65%+ inlier ratio
        "LC_MAX_CAND":      15,
        "LC_MIN_SEP":       80,
        "LC_MAX_CORRECTION":  3.0,
        "LC_MAX_YAW_DEG":   35.0,
        "LC_ICP_MAX_DIST":  8.0,
        "LC_ICP_MAX_ITER":  100,
        "LC_ICP_VOXEL":     0.50,
        "LC_FITNESS_DIST":  1.0,        # [FIX C] tight inlier distance
        "LC_PGO_BATCH":     3,          # accumulate 3 loops before PGO
        "LC_PGO_COOLDOWN":  15,         # kf to skip LC after each PGO
        "LC_HIST_DEDUP":    30,         # min kf gap before same hist_idx reused
    },
[FINAL STATS]
  Target:                     Track3
  Total frames:               1738
  Keyframes:                  314
  Loop closures applied:      6
  ICP success:                1589/1737 (91.5%)
  ICP reject (score):         146
  ICP reject (delta_trans):   2
  ICP fail (other):           0
  LC ICP total:               0
  LC ICP success:             0/0 (0.0%)
  LC ICP fail:                0
  Start->End distance:        9.459 m
  Cumulative path length:     435.723 m
  Max single-frame movement:  2.736 m

── Proxy metrics ──────────────────────────────────────
  Drift rate  (end/path):     0.0217  △ moderate
  Closure gap (start→end):    9.459 m  ✗
  RMS jerk    (global):       0.1041 m/f²
  RMS jerk    per quarter:
    Q1 (0-25%):  0.0251 m/f²  ✓
    Q2 (25-50%):  0.0573 m/f²  △
    Q3 (50-75%):  0.0435 m/f²  ✓
    Q4 (75-100%):  0.1938 m/f²  ✗ HIGH
  Local shape deviation:
    max:  2.445 m  △
    p95:  1.651 m
  
//main_track2_lc.py 純 odom 沒有 loopclosure
ICP_VOXEL = 0.25
ICP_MAX_DIST = 2.5
ICP_REJECT_SCORE = 3.0
ICP_MAX_ITER = 50
MAX_DELTA_TRANS = 10.0
KEYFRAME_DIST = 1.0
KEYFRAME_ANG = 10
SUBMAP_WINDOW = 3
ENABLE_LC = 0

output:
kaggle score= 4.9m

//main_track2_lc.py 純 odom 沒有 loopclosure
[PARAM] ICP_VOXEL = 0.3
[PARAM] ICP_MAX_DIST = 3.0
[PARAM] ICP_REJECT_SCORE = 3.0
[PARAM] ICP_MAX_ITER = 50
[PARAM] MAX_DELTA_TRANS = 10.0
[PARAM] KEYFRAME_DIST = 1.0
[PARAM] KEYFRAME_ANG = 10.0
[PARAM] SUBMAP_WINDOW = 5
[PARAM] ENABLE_LC = False

output:
kaggle score= 20m

//有 LC
[PARAM] ICP_VOXEL = 0.25
[PARAM] ICP_MAX_DIST = 2.5
[PARAM] ICP_REJECT_SCORE = 3.0
[PARAM] ICP_MAX_ITER = 50
[PARAM] MAX_DELTA_TRANS = 10.0
[PARAM] KEYFRAME_DIST = 1.0
[PARAM] KEYFRAME_ANG = 10.0
[PARAM] SUBMAP_WINDOW = 3
[PARAM] ENABLE_LC = True
[PARAM] LC_DIST = 10.0
[PARAM] LC_SCORE = 0.08
[PARAM] LC_MAX_CAND = 15
[PARAM] LC_MIN_SEP = 80
[PARAM] LC_MAX_CORRECTION = 8.0
[PARAM] LC_MAX_YAW_DEG = 30.0

output:
kaggle score= 4.9m

//LC and score=0.06
[PARAM] ICP_VOXEL = 0.25
[PARAM] ICP_MAX_DIST = 2.5
[PARAM] ICP_REJECT_SCORE = 3.0
[PARAM] ICP_MAX_ITER = 50
[PARAM] MAX_DELTA_TRANS = 10.0
[PARAM] KEYFRAME_DIST = 1.0
[PARAM] KEYFRAME_ANG = 10.0
[PARAM] SUBMAP_WINDOW = 3
[PARAM] ENABLE_LC = True
[PARAM] LC_DIST = 8.0
[PARAM] LC_SCORE = 0.06
[PARAM] LC_MAX_CAND = 15
[PARAM] LC_MIN_SEP = 80
[PARAM] LC_MAX_CORRECTION = 3.0
[PARAM] LC_MAX_YAW_DEG = 30.0


//LC_MAX_CAND + LC_MAX_CORRECTION
[PARAM] ICP_VOXEL = 0.25
[PARAM] ICP_MAX_DIST = 2.5
[PARAM] ICP_REJECT_SCORE = 3.0
[PARAM] ICP_MAX_ITER = 50
[PARAM] MAX_DELTA_TRANS = 10.0
[PARAM] KEYFRAME_DIST = 1.0
[PARAM] KEYFRAME_ANG = 10.0
[PARAM] SUBMAP_WINDOW = 3
[PARAM] ENABLE_LC = True
[PARAM] LC_DIST = 20.0
[PARAM] LC_SCORE = 0.06
[PARAM] LC_MAX_CAND = 10
[PARAM] LC_MIN_SEP = 80
[PARAM] LC_MAX_CORRECTION = 10.0
[PARAM] LC_MAX_YAW_DEG = 30.0




[STATS] frame 50: pose=(-0.68,-5.12,-0.12) cum=5.46m avg50=0.1114m/f
[STATS] frame 100: pose=(-0.94,-12.78,-0.12) cum=13.65m avg50=0.1639m/f
[STATS] frame 150: pose=(-1.29,-20.56,0.04) cum=21.69m avg50=0.1607m/f
[STATS] frame 200: pose=(3.06,-24.92,0.33) cum=29.19m avg50=0.1500m/f
[STATS] frame 250: pose=(14.20,-25.82,0.76) cum=40.42m avg50=0.2246m/f
[STATS] frame 300: pose=(27.45,-26.55,1.21) cum=53.77m avg50=0.2670m/f
[STATS] frame 350: pose=(41.29,-26.96,1.70) cum=67.74m avg50=0.2795m/f
[STATS] frame 400: pose=(54.82,-27.26,2.24) cum=81.41m avg50=0.2734m/f
[STATS] frame 450: pose=(68.50,-27.69,2.74) cum=95.17m avg50=0.2751m/f
[STATS] frame 500: pose=(81.88,-28.17,3.35) cum=108.69m avg50=0.2704m/f
[STATS] frame 550: pose=(90.87,-21.55,3.50) cum=121.38m avg50=0.2537m/f
[STATS] frame 600: pose=(91.23,-6.04,2.68) cum=137.04m avg50=0.3133m/f
[STATS] frame 650: pose=(89.23,7.54,1.77) cum=151.72m avg50=0.2937m/f
[STATS] frame 700: pose=(77.29,7.23,1.62) cum=163.88m avg50=0.2431m/f
[STATS] frame 750: pose=(74.88,-2.65,2.19) cum=174.61m avg50=0.2148m/f
[STATS] frame 800: pose=(62.25,-2.74,2.23) cum=188.94m avg50=0.2864m/f
[STATS] frame 850: pose=(46.59,-2.73,1.72) cum=205.43m avg50=0.3300m/f
[STATS] frame 900: pose=(30.68,-1.60,1.59) cum=222.13m avg50=0.3340m/f
[STATS] frame 950: pose=(25.30,9.94,6.12) cum=239.00m avg50=0.3373m/f
[STATS] frame 1000: pose=(26.62,25.94,13.13) cum=256.55m avg50=0.3510m/f
[STATS] frame 1050: pose=(27.14,40.94,19.75) cum=272.98m avg50=0.3287m/f
[STATS] frame 1100: pose=(37.92,45.16,22.51) cum=287.65m avg50=0.2935m/f
[STATS] frame 1150: pose=(52.68,44.59,23.85) cum=302.81m avg50=0.3031m/f
[STATS] frame 1200: pose=(58.69,37.27,20.51) cum=315.21m avg50=0.2481m/f
[STATS] frame 1250: pose=(61.49,32.22,18.10) cum=323.19m avg50=0.1596m/f
[STATS] frame 1300: pose=(64.89,21.06,12.53) cum=337.40m avg50=0.2842m/f
[STATS] frame 1350: pose=(56.68,15.18,8.48) cum=351.74m avg50=0.2869m/f
[STATS] frame 1400: pose=(42.58,19.45,9.30) cum=366.55m avg50=0.2961m/f
[STATS] frame 1450: pose=(33.00,22.33,9.99) cum=377.82m avg50=0.2254m/f
[STATS] frame 1500: pose=(26.75,11.41,3.41) cum=392.87m avg50=0.3011m/f
[STATS] frame 1550: pose=(20.02,2.32,-2.50) cum=408.17m avg50=0.3060m/f
[STATS] frame 1600: pose=(4.42,4.62,-2.87) cum=424.18m avg50=0.3201m/f
[STATS] frame 1650: pose=(0.58,5.37,-1.89) cum=433.33m avg50=0.1830m/f
[STATS] frame 1700: pose=(0.66,5.48,-3.11) cum=436.01m avg50=0.0537m/f

// last
[INFO] Processing frame 1/1738
[INFO] Initial frame added.
[WARN] ICP rejected or failed, use predicted pose.
[WARN] ICP rejected or failed, use predicted pose.
[INFO] Processing frame 50/1738
[INFO] Add keyframe 10, frame=69, rel_trans=1.094
[INFO] Processing frame 100/1738
[INFO] Processing frame 150/1738
[INFO] Add keyframe 20, frame=150, rel_trans=1.090
[INFO] Processing frame 200/1738
[INFO] Add keyframe 30, frame=202, rel_trans=1.106
[INFO] Processing frame 250/1738
[INFO] Add keyframe 40, frame=252, rel_trans=1.193
[INFO] Add keyframe 50, frame=294, rel_trans=1.206
[INFO] Processing frame 300/1738
[INFO] Add keyframe 60, frame=335, rel_trans=1.163
[INFO] Processing frame 350/1738
[INFO] Add keyframe 70, frame=375, rel_trans=1.068
[INFO] Processing frame 400/1738
[INFO] Add keyframe 80, frame=416, rel_trans=1.145
[INFO] Processing frame 450/1738
[INFO] Add keyframe 90, frame=457, rel_trans=1.066
[INFO] Processing frame 500/1738
[LC_DEBUG] kf=100, candidates=0, nearest_hist_dist=83.80
[INFO] Add keyframe 100, frame=500, rel_trans=1.259
[INFO] Add keyframe 110, frame=544, rel_trans=0.930
[INFO] Processing frame 550/1738
[LC_DEBUG] kf=120, candidates=0, nearest_hist_dist=78.78
[INFO] Add keyframe 120, frame=584, rel_trans=1.026
[INFO] Processing frame 600/1738
[INFO] Add keyframe 130, frame=619, rel_trans=1.276
[INFO] Processing frame 650/1738
[LC_DEBUG] kf=140, candidates=0, nearest_hist_dist=62.81
[INFO] Add keyframe 140, frame=652, rel_trans=0.461
[INFO] Add keyframe 150, frame=693, rel_trans=0.733
[INFO] Processing frame 700/1738
[LC_DEBUG] kf=160, candidates=0, nearest_hist_dist=34.08
[INFO] Add keyframe 160, frame=729, rel_trans=1.130
[INFO] Processing frame 750/1738
[INFO] Add keyframe 170, frame=764, rel_trans=1.093
[WARN] ICP rejected or failed, use predicted pose.
[WARN] ICP rejected or failed, use predicted pose.
[INFO] Processing frame 800/1738
[LC_DEBUG] kf=180, candidates=0, nearest_hist_dist=25.03
[INFO] Add keyframe 180, frame=804, rel_trans=1.174
[INFO] Add keyframe 190, frame=840, rel_trans=1.055
[WARN] ICP rejected or failed, use predicted pose.
[INFO] Processing frame 850/1738
[LC_DEBUG] kf=200, candidates=0, nearest_hist_dist=24.72
[INFO] Add keyframe 200, frame=878, rel_trans=1.021
[INFO] Processing frame 900/1738
[INFO] Add keyframe 210, frame=913, rel_trans=1.277
[LC_DEBUG] kf=220, candidates=0, nearest_hist_dist=25.12
[INFO] Add keyframe 220, frame=939, rel_trans=1.026
[INFO] Processing frame 950/1738
[INFO] Add keyframe 230, frame=970, rel_trans=1.000
[INFO] Processing frame 1000/1738
[LC_DEBUG] kf=240, candidates=0, nearest_hist_dist=37.55
[INFO] Add keyframe 240, frame=1001, rel_trans=1.319
[INFO] Add keyframe 250, frame=1038, rel_trans=1.321
[INFO] Processing frame 1050/1738
[LC_DEBUG] kf=260, candidates=0, nearest_hist_dist=55.09
[INFO] Add keyframe 260, frame=1066, rel_trans=0.573
[INFO] Processing frame 1100/1738
[INFO] Add keyframe 270, frame=1103, rel_trans=1.041
[LC_DEBUG] kf=280, candidates=0, nearest_hist_dist=49.94
[INFO] Add keyframe 280, frame=1141, rel_trans=1.248
[INFO] Processing frame 1150/1738
[INFO] Add keyframe 290, frame=1186, rel_trans=1.226
[INFO] Processing frame 1200/1738
[LC_DEBUG] kf=300, candidates=0, nearest_hist_dist=35.24
[INFO] Add keyframe 300, frame=1233, rel_trans=0.298
[INFO] Processing frame 1250/1738
[INFO] Add keyframe 310, frame=1262, rel_trans=0.901
[INFO] Processing frame 1300/1738
[LC_DEBUG] kf=320, candidates=0, nearest_hist_dist=20.36
[INFO] Add keyframe 320, frame=1301, rel_trans=1.077
[INFO] Add keyframe 330, frame=1332, rel_trans=0.942
[INFO] Processing frame 1350/1738
[LC_DEBUG] kf=340, candidates=10, nearest_hist_dist=19.23
[INFO] Add keyframe 340, frame=1372, rel_trans=1.254
[INFO] Processing frame 1400/1738
[INFO] Add keyframe 350, frame=1410, rel_trans=0.976
[LC_DEBUG] kf=360, candidates=10, nearest_hist_dist=6.81
[INFO] Add keyframe 360, frame=1444, rel_trans=0.632
[INFO] Processing frame 1450/1738
[INFO] Add keyframe 370, frame=1471, rel_trans=1.201
[INFO] Processing frame 1500/1738
[WARN] ICP rejected or failed, use predicted pose.
[WARN] ICP rejected or failed, use predicted pose.
[LC_DEBUG] kf=380, candidates=10, nearest_hist_dist=2.25
[INFO] Add keyframe 380, frame=1510, rel_trans=1.261
[INFO] Add keyframe 390, frame=1539, rel_trans=0.595
[INFO] Processing frame 1550/1738
[LC_DEBUG] kf=400, candidates=10, nearest_hist_dist=12.03
[INFO] Add keyframe 400, frame=1579, rel_trans=1.264
[INFO] Processing frame 1600/1738
[INFO] Add keyframe 410, frame=1616, rel_trans=0.573
[INFO] Processing frame 1650/1738
[INFO] Processing frame 1700/1738