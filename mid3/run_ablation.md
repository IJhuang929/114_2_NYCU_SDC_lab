============================================================
SEQ_1 ABLATIONS
============================================================

>>> [seq1_full] running seq=seq_1 ...
============================================================
Run tracking (seq_1/seq_2 mode) — seq_1
============================================================
frames: 489, output: ../outputs/ablation/seq1_full/seq_1_student_20260610_142318
max_age: 8, min_hits: 1, max_distance: 5.0
  frame 0000: detections=0, confirmed=0
  frame 0050: detections=0, confirmed=0
  frame 0100: detections=13, confirmed=13
  frame 0150: detections=7, confirmed=7
  frame 0200: detections=5, confirmed=5
  frame 0250: detections=11, confirmed=11
  frame 0300: detections=4, confirmed=4
  frame 0350: detections=6, confirmed=6
  frame 0400: detections=8, confirmed=8
  frame 0450: detections=8, confirmed=8
[INFO] offline stitching remap:
  1 -> 0
  2 -> 0
  3 -> 0
  4 -> 0
  5 -> 0
  6 -> 0
  7 -> 0
  8 -> 0
  9 -> 0
  10 -> 0
  11 -> 0
  12 -> 0
  13 -> 0
  14 -> 0
  15 -> 0
  16 -> 0
  19 -> 0
  20 -> 0
  21 -> 0
  22 -> 0
  23 -> 0
  24 -> 0
  25 -> 0
  26 -> 0
  27 -> 0
  28 -> 0
  29 -> 0
  30 -> 0
  31 -> 0
  32 -> 0
  33 -> 0
  34 -> 0
  35 -> 0
  36 -> 0
  38 -> 0
  39 -> 0
  40 -> 0
  41 -> 0
  42 -> 0
  43 -> 0
  44 -> 0
  45 -> 0
  46 -> 0
  47 -> 0
  48 -> 37
  49 -> 0
  50 -> 0
  51 -> 0
  52 -> 0
  53 -> 0
  54 -> 0
  55 -> 0
  56 -> 0
  57 -> 0
  58 -> 0
  59 -> 0
  60 -> 0
  61 -> 0
[INFO] tracklets: 62
[INFO] stitched ids: 58
[INFO] swap corrections: 0
[DONE] output folder: ../outputs/ablation/seq1_full/seq_1_student_20260610_142318
[DONE] result csv: ../outputs/ablation/seq1_full/seq_1_student_20260610_142318/result.csv
 frames  gt_positive_points  FP_points  FN_points  IDSW_proxy  fragmentation_proxy  MOTA_proxy  MOTA_style_score
    489               10058          0          0           2                  409    0.999801          0.999801
[DONE] saved: ../outputs/ablation/seq1_full_score.csv

  ┌─ seq1_full ───────────────────────────────
  │  frames                        489.0000
  │  gt_positive_points          10058.0000
  │  FP_points                            0
  │  FN_points                            0
  │  IDSW_proxy                           2
  │  fragmentation_proxy                409
  │  MOTA_proxy                  [███████████████████░] 0.9998
  │  MOTA_style_score                0.9998
  └────────────────────────────────────────────────

  >>> [seq1_no_offline] running seq=seq_1 ...
============================================================
Run tracking (seq_1/seq_2 mode) — seq_1
============================================================
frames: 489, output: ../outputs/ablation/seq1_no_offline/seq_1_student_20260610_142527
max_age: 8, min_hits: 1, max_distance: 5.0
  frame 0000: detections=0, confirmed=0
  frame 0050: detections=0, confirmed=0
  frame 0100: detections=13, confirmed=13
  frame 0150: detections=7, confirmed=7
  frame 0200: detections=5, confirmed=5
  frame 0250: detections=11, confirmed=11
  frame 0300: detections=4, confirmed=4
  frame 0350: detections=6, confirmed=6
  frame 0400: detections=8, confirmed=8
  frame 0450: detections=8, confirmed=8
[INFO] tracklets: 62
[INFO] stitched ids: 0
[INFO] swap corrections: 0
[DONE] output folder: ../outputs/ablation/seq1_no_offline/seq_1_student_20260610_142527
[DONE] result csv: ../outputs/ablation/seq1_no_offline/seq_1_student_20260610_142527/result.csv
 frames  gt_positive_points  FP_points  FN_points  IDSW_proxy  fragmentation_proxy  MOTA_proxy  MOTA_style_score
    489               10058          0          0          49                  409    0.995128          0.995128
[DONE] saved: ../outputs/ablation/seq1_no_offline_score.csv

  ┌─ seq1_no_offline ───────────────────────────────
  │  frames                        489.0000
  │  gt_positive_points          10058.0000
  │  FP_points                            0
  │  FN_points                            0
  │  IDSW_proxy                          49
  │  fragmentation_proxy                409
  │  MOTA_proxy                  [███████████████████░] 0.9951
  │  MOTA_style_score                0.9951
  └────────────────────────────────────────────────
 patch_kf_predict: OK

>>> [seq1_kf_predict] running seq=seq_1 ...
============================================================
Run tracking (seq_1/seq_2 mode) — seq_1
============================================================
frames: 489, output: ../outputs/ablation/seq1_kf_predict/seq_1_student_20260610_142737
max_age: 8, min_hits: 1, max_distance: 5.0
  frame 0000: detections=0, confirmed=0
  frame 0050: detections=0, confirmed=0
  frame 0100: detections=13, confirmed=13
  frame 0150: detections=7, confirmed=7
  frame 0200: detections=5, confirmed=5
  frame 0250: detections=11, confirmed=11
  frame 0300: detections=4, confirmed=4
  frame 0350: detections=6, confirmed=6
  frame 0400: detections=8, confirmed=8
  frame 0450: detections=8, confirmed=8
[INFO] offline stitching remap:
  1 -> 0
  2 -> 0
  3 -> 0
  4 -> 0
  5 -> 0
  6 -> 0
  7 -> 0
  8 -> 0
  9 -> 0
  10 -> 0
  11 -> 0
  12 -> 0
  13 -> 0
  14 -> 0
  15 -> 0
  16 -> 0
  17 -> 0
  18 -> 0
  19 -> 0
  20 -> 0
  21 -> 0
  22 -> 0
  23 -> 0
  24 -> 0
  25 -> 0
  26 -> 0
  27 -> 0
  28 -> 0
  29 -> 0
  30 -> 0
  31 -> 0
  32 -> 0
  33 -> 0
  34 -> 0
  35 -> 0
  36 -> 0
  37 -> 0
  38 -> 0
  39 -> 0
  40 -> 0
  41 -> 0
  42 -> 0
  43 -> 0
  44 -> 0
  45 -> 0
  46 -> 0
  47 -> 0
  48 -> 0
  49 -> 0
  50 -> 0
  51 -> 0
  52 -> 0
  53 -> 0
  54 -> 0
  55 -> 0
  56 -> 0
  57 -> 0
  58 -> 0
  59 -> 0
  60 -> 0
  61 -> 0
  62 -> 0
  63 -> 0
  64 -> 0
  65 -> 0
  66 -> 0
  67 -> 0
  68 -> 0
  69 -> 0
  70 -> 0
  71 -> 0
  72 -> 0
  73 -> 0
  74 -> 0
  75 -> 0
  76 -> 0
  77 -> 0
  78 -> 0
  79 -> 0
  80 -> 0
  81 -> 0
  82 -> 0
  83 -> 0
  84 -> 0
  85 -> 0
  86 -> 0
  87 -> 0
  88 -> 0
  89 -> 0
  90 -> 0
  91 -> 0
  92 -> 0
  93 -> 0
  94 -> 0
  95 -> 0
  96 -> 0
  97 -> 0
  98 -> 0
  99 -> 0
  100 -> 0
  101 -> 0
  102 -> 0
  103 -> 0
  104 -> 0
  105 -> 0
  106 -> 0
  107 -> 0
  108 -> 0
  109 -> 0
  110 -> 0
  111 -> 0
  112 -> 0
  113 -> 0
  114 -> 0
  115 -> 0
  116 -> 0
  117 -> 0
  118 -> 0
  119 -> 0
  120 -> 0
  121 -> 0
  122 -> 0
  123 -> 0
  124 -> 0
  125 -> 0
  126 -> 0
  127 -> 0
  128 -> 0
  129 -> 0
  130 -> 0
  131 -> 0
  132 -> 0
  133 -> 0
  134 -> 0
  135 -> 0
  136 -> 0
  137 -> 0
  138 -> 0
  139 -> 0
  140 -> 0
  141 -> 0
  142 -> 0
  143 -> 0
  144 -> 0
  145 -> 0
  146 -> 0
  147 -> 0
  148 -> 0
  149 -> 0
  150 -> 0
  151 -> 0
  152 -> 0
  153 -> 0
  154 -> 0
  155 -> 0
  156 -> 0
  157 -> 0
  158 -> 0
  159 -> 0
  160 -> 0
  161 -> 0
  162 -> 0
  163 -> 0
[INFO] tracklets: 164
[INFO] stitched ids: 163
[INFO] swap corrections: 0
[DONE] output folder: ../outputs/ablation/seq1_kf_predict/seq_1_student_20260610_142737
[DONE] result csv: ../outputs/ablation/seq1_kf_predict/seq_1_student_20260610_142737/result.csv
 frames  gt_positive_points  FP_points  FN_points  IDSW_proxy  fragmentation_proxy  MOTA_proxy  MOTA_style_score
    489               10058          0          0           0                  409         1.0               1.0
[DONE] saved: ../outputs/ablation/seq1_kf_predict_score.csv

  ┌─ seq1_kf_predict ───────────────────────────────
  │  frames                        489.0000
  │  gt_positive_points          10058.0000
  │  FP_points                            0
  │  FN_points                            0
  │  IDSW_proxy                           0
  │  fragmentation_proxy                409
  │  MOTA_proxy                  [████████████████████] 1.0000
  │  MOTA_style_score                1.0000
  └────────────────────────────────────────────────

  ============================================================
SEQ_3 ABLATIONS
============================================================

>>> [seq3_full] running seq=seq_3 ...
ego_global_pos: ../public_dev/seq_3/ego_global_pos.txt
ego motion steps loaded: 599
ego compensation is enabled.
ego_delta[0] = [2.64012526e-01 1.38777878e-17]
============================================================
Run tracking (seq_3/seq_4 mode) — seq_3
============================================================
frames: 600, output: ../outputs/ablation/seq3_full/seq_3_student_20260610_142946
max_age: 8, min_hits: 1, max_distance: 5.0
  frame 0000: detections=8, confirmed=8
ego_motion[0] = [2.64012526e-01 1.38777878e-17]
  frame 0050: detections=5, confirmed=5
  frame 0100: detections=4, confirmed=4
  frame 0150: detections=4, confirmed=4
  frame 0200: detections=3, confirmed=3
  frame 0250: detections=7, confirmed=7
  frame 0300: detections=3, confirmed=3
  frame 0350: detections=4, confirmed=4
  frame 0400: detections=6, confirmed=6
  frame 0450: detections=6, confirmed=6
  frame 0500: detections=3, confirmed=3
  frame 0550: detections=4, confirmed=4
[DONE] output folder: ../outputs/ablation/seq3_full/seq_3_student_20260610_142946
[DONE] result csv: ../outputs/ablation/seq3_full/seq_3_student_20260610_142946/result.csv
 frames  gt_positive_points  FP_points  FN_points  IDSW_proxy  fragmentation_proxy  MOTA_proxy  MOTA_style_score
    600                9688          0          0          31                  356      0.9968            0.9968
[DONE] saved: ../outputs/ablation/seq3_full_score.csv

  ┌─ seq3_full ───────────────────────────────
  │  frames                        600.0000
  │  gt_positive_points           9688.0000
  │  FP_points                            0
  │  FN_points                            0
  │  IDSW_proxy                          31
  │  fragmentation_proxy                356
  │  MOTA_proxy                  [███████████████████░] 0.9968
  │  MOTA_style_score                0.9968
  └────────────────────────────────────────────────

>>> [seq3_no_ego] running seq=seq_3 ...
ego_global_pos: ../public_dev/seq_3/ego_global_pos.txt
ego motion steps loaded: 599
ego compensation is enabled.
ego_delta[0] = [2.64012526e-01 1.38777878e-17]
============================================================
Run tracking (seq_3/seq_4 mode) — seq_3
============================================================
frames: 600, output: ../outputs/ablation/seq3_no_ego/seq_3_student_20260610_143218
max_age: 8, min_hits: 1, max_distance: 5.0
  frame 0000: detections=8, confirmed=8
ego_motion[0] = None
  frame 0050: detections=5, confirmed=5
  frame 0100: detections=4, confirmed=4
  frame 0150: detections=4, confirmed=4
  frame 0200: detections=3, confirmed=3
  frame 0250: detections=7, confirmed=7
  frame 0300: detections=3, confirmed=3
  frame 0350: detections=4, confirmed=4
  frame 0400: detections=6, confirmed=6
  frame 0450: detections=6, confirmed=6
  frame 0500: detections=3, confirmed=3
  frame 0550: detections=4, confirmed=4
[DONE] output folder: ../outputs/ablation/seq3_no_ego/seq_3_student_20260610_143218
[DONE] result csv: ../outputs/ablation/seq3_no_ego/seq_3_student_20260610_143218/result.csv
 frames  gt_positive_points  FP_points  FN_points  IDSW_proxy  fragmentation_proxy  MOTA_proxy  MOTA_style_score
    600                9688          0          0          65                  356    0.993291          0.993291
[DONE] saved: ../outputs/ablation/seq3_no_ego_score.csv

  ┌─ seq3_no_ego ───────────────────────────────
  │  frames                        600.0000
  │  gt_positive_points           9688.0000
  │  FP_points                            0
  │  FN_points                            0
  │  IDSW_proxy                          65
  │  fragmentation_proxy                356
  │  MOTA_proxy                  [███████████████████░] 0.9933
  │  MOTA_style_score                0.9933
  └────────────────────────────────────────────────
   patch_with_kf: kf already enabled in new-track init — no patch needed
  main_with_kf written to /tmp/main_with_kf.py (replaced=True)

>>> [seq3_with_kf] running seq=seq_3 ...
ego_global_pos: ../public_dev/seq_3/ego_global_pos.txt
ego motion steps loaded: 599
ego compensation is enabled.
ego_delta[0] = [2.64012526e-01 1.38777878e-17]
============================================================
Run tracking (seq_3/seq_4 mode) — seq_3
============================================================
frames: 600, output: ../outputs/ablation/seq3_with_kf/seq_3_student_20260610_143451
max_age: 8, min_hits: 1, max_distance: 5.0
  frame 0000: detections=8, confirmed=8
ego_motion[0] = [2.64012526e-01 1.38777878e-17]
  frame 0050: detections=5, confirmed=5
  frame 0100: detections=4, confirmed=4
  frame 0150: detections=4, confirmed=4
  frame 0200: detections=3, confirmed=3
  frame 0250: detections=7, confirmed=7
  frame 0300: detections=3, confirmed=3
  frame 0350: detections=4, confirmed=4
  frame 0400: detections=6, confirmed=6
  frame 0450: detections=6, confirmed=6
  frame 0500: detections=3, confirmed=3
  frame 0550: detections=4, confirmed=4
[DONE] output folder: ../outputs/ablation/seq3_with_kf/seq_3_student_20260610_143451
[DONE] result csv: ../outputs/ablation/seq3_with_kf/seq_3_student_20260610_143451/result.csv
 frames  gt_positive_points  FP_points  FN_points  IDSW_proxy  fragmentation_proxy  MOTA_proxy  MOTA_style_score
    600                9688          0          0          31                  356      0.9968            0.9968
[DONE] saved: ../outputs/ablation/seq3_with_kf_score.csv

  ┌─ seq3_with_kf ───────────────────────────────
  │  frames                        600.0000
  │  gt_positive_points           9688.0000
  │  FP_points                            0
  │  FN_points                            0
  │  IDSW_proxy                          31
  │  fragmentation_proxy                356
  │  MOTA_proxy                  [███████████████████░] 0.9968
  │  MOTA_style_score                0.9968
  └────────────────────────────────────────────────
  vis → ../outputs/ablation/seq3_with_kf/seq_3_student_20260610_143451/tracking_vis  (600 frames)

============================================================
FINAL COMPARISON TABLE
============================================================
  Experiment                 MOTA_proxy      FP_points      FN_points     IDSW_proxy  fragmentation
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  seq1_full                      0.9998              0              0              2            409
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  seq1_no_offline                0.9951              0              0             49            409
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  seq1_kf_predict                1.0000              0              0              0            409
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  seq3_full                      0.9968              0              0             31            356
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  seq3_no_ego                    0.9933              0              0             65            356
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  seq3_with_kf                   0.9968              0              0             31            356