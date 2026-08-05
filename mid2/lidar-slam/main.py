import os
import numpy as np
import pandas as pd
import open3d as o3d
import small_gicp as sg

from scipy.spatial.transform import Rotation as ScipyR
from scipy.spatial.transform import Slerp

from utils.ICP import ICP
from utils.submap import SubmapManager
from utils.keyframe import KeyframeManager
from utils.posegraph import PoseGraph
from utils.io import load_folder, load_pcd
from utils.viz import Visualizer


# ============================================================
# Path setting
# ============================================================

target = os.environ.get("TARGET", "Track2")
dataset_root = os.environ.get("MID2_DATASET_ROOT", "mid2dataset")

pcd_folder = os.path.join(dataset_root, target, "data", "raw_pcd")
output_csv = os.path.join(dataset_root, target, "result","lidar_poses.csv")

files = load_folder(pcd_folder)

print(f"[INFO] Target:            {target}")
print(f"[INFO] PCD folder:        {pcd_folder}")
print(f"[INFO] Number of frames:  {len(files)}")
print(f"[INFO] Output CSV:        {output_csv}")

if len(files) == 0:
    raise RuntimeError(f"No PCD files found in {pcd_folder}")


# ============================================================
# Per-track default parameters
# ============================================================

_TRACK_DEFAULTS = {
    "Track1": {
        "ICP_VOXEL":        0.10,
        "ICP_MAX_DIST":     2.0,
        "ICP_MAX_ITER":     100,
        "MAX_DELTA_TRANS":  5.0,
        "ICP_REJECT_SCORE": 2.5,
        "KEYFRAME_DIST":    0.5,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    8,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        0,
        "LC_DIST":          6.0,
        "LC_SCORE":         0.08,
        "LC_MAX_CAND":      15,
        "LC_MIN_SEP":       50,
        "LC_MAX_CORRECTION":  4.0,
        "LC_MAX_YAW_DEG":   30.0,
        "LC_ICP_MAX_DIST":  None,
        "LC_ICP_MAX_ITER":  None,
        "LC_ICP_VOXEL":     None,
        "LC_FITNESS_DIST":  None,
    },
    "Track2": {
        "ICP_VOXEL":        0.25,
        "ICP_MAX_DIST":     2.5,
        "ICP_MAX_ITER":     50,
        "MAX_DELTA_TRANS":  5.0,
        "ICP_REJECT_SCORE": 1.0,
        "KEYFRAME_DIST":    1.0,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    3,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        0,
        "LC_DIST":          8.0,   
        "LC_SCORE":         0.06,   
        "LC_MAX_CAND":      15,     
        "LC_MIN_SEP":       80,    
        "LC_MAX_CORRECTION":  3.0,  
        "LC_MAX_YAW_DEG":   25.0,
        "LC_ICP_MAX_DIST":  None,
        "LC_ICP_MAX_ITER":  None,
        "LC_ICP_VOXEL":     None,
        "LC_FITNESS_DIST":  None,
    },
    "Track3": {
        "ICP_VOXEL":        0.10,
        "ICP_MAX_DIST":     2.5,
        "ICP_MAX_ITER":     80,
        "MAX_DELTA_TRANS":  10.0,
        "ICP_REJECT_SCORE": 3.0,
        "KEYFRAME_DIST":    1.0,
        "KEYFRAME_ANG":     10.0,
        "SUBMAP_WINDOW":    14,
        "SUBMAP_VOXEL":     0,
        "ENABLE_LC":        1,
        "LC_DIST":          8.0,
        "LC_SCORE":         0.4,
        "LC_MAX_CAND":      6,
        "LC_MIN_SEP":       80,
        "LC_MAX_CORRECTION":  8.0,
        "LC_MAX_YAW_DEG":   45.0,
        # Track3-specific (loop-closure rotation gate + cooldown + coarse ICP)
        "LC_MAX_ROT_DEG":      20.0,
        "LC_COOLDOWN":         10,
        "LC_COARSE_VOXEL":     0.5,
        "LC_COARSE_MAX_DIST":  10.0,
        "LC_COARSE_ITER":      30,
        "LC_COARSE_REJECT":    5.0,
        "LC_COARSE_MAX_DELTA": 12.0,
    },
}

_d = _TRACK_DEFAULTS.get(target, _TRACK_DEFAULTS["Track2"])
IS_TRACK3 = (target == "Track3")

ICP_VOXEL        = float(os.environ.get("ICP_VOXEL",        str(_d["ICP_VOXEL"])))
ICP_MAX_DIST     = float(os.environ.get("ICP_MAX_DIST",     str(_d["ICP_MAX_DIST"])))
ICP_REJECT_SCORE = float(os.environ.get("ICP_REJECT_SCORE", str(_d["ICP_REJECT_SCORE"])))
ICP_MAX_ITER     = int  (os.environ.get("ICP_MAX_ITER",     str(_d["ICP_MAX_ITER"])))
MAX_DELTA_TRANS  = float(os.environ.get("MAX_DELTA_TRANS",  str(_d["MAX_DELTA_TRANS"])))

KEYFRAME_DIST  = float(os.environ.get("KEYFRAME_DIST",  str(_d["KEYFRAME_DIST"])))
KEYFRAME_ANG   = float(os.environ.get("KEYFRAME_ANG",   str(_d["KEYFRAME_ANG"])))
SUBMAP_WINDOW  = int  (os.environ.get("SUBMAP_WINDOW",  str(_d["SUBMAP_WINDOW"])))
SUBMAP_VOXEL   = float(os.environ.get("SUBMAP_VOXEL",   str(_d["SUBMAP_VOXEL"])))

ENABLE_LC          = os.environ.get("ENABLE_LC", "1" if _d["ENABLE_LC"] else "0") == "1"
LC_DIST            = float(os.environ.get("LC_DIST",            str(_d["LC_DIST"])))
LC_SCORE           = float(os.environ.get("LC_SCORE",           str(_d["LC_SCORE"])))
LC_MAX_CAND        = int  (os.environ.get("LC_MAX_CAND",        str(_d["LC_MAX_CAND"])))
LC_MIN_SEP         = int  (os.environ.get("LC_MIN_SEP",         str(_d["LC_MIN_SEP"])))
LC_MAX_CORRECTION  = float(os.environ.get("LC_MAX_CORRECTION",  str(_d["LC_MAX_CORRECTION"])))
LC_MAX_YAW_DEG     = float(os.environ.get("LC_MAX_YAW_DEG",     str(_d["LC_MAX_YAW_DEG"])))

def _lc_icp_default(key, odom_val):
    track_val = _d.get(key)
    base = odom_val if track_val is None else track_val
    env_val = os.environ.get(key)
    return float(env_val) if env_val is not None else base

LC_ICP_MAX_DIST  = _lc_icp_default("LC_ICP_MAX_DIST", ICP_MAX_DIST)
LC_ICP_MAX_ITER  = int(_lc_icp_default("LC_ICP_MAX_ITER", ICP_MAX_ITER))
LC_ICP_VOXEL     = _lc_icp_default("LC_ICP_VOXEL",    ICP_VOXEL)
LC_FITNESS_DIST  = _lc_icp_default("LC_FITNESS_DIST",  1.0)

def _lc_int(key, default=1):
    return int(os.environ.get(key, str(int(_d.get(key, default)))))

LC_PGO_BATCH    = _lc_int("LC_PGO_BATCH",    1)
LC_PGO_COOLDOWN = _lc_int("LC_PGO_COOLDOWN", 10)
LC_HIST_DEDUP   = _lc_int("LC_HIST_DEDUP",   20)

# Track3-specific parameters (only meaningful when target == "Track3")
if IS_TRACK3:
    LC_MAX_ROT_DEG      = float(os.environ.get("LC_MAX_ROT_DEG",      str(_d["LC_MAX_ROT_DEG"])))
    LC_COOLDOWN         = int  (os.environ.get("LC_COOLDOWN",         str(_d["LC_COOLDOWN"])))
    LC_COARSE_VOXEL     = float(os.environ.get("LC_COARSE_VOXEL",     str(_d["LC_COARSE_VOXEL"])))
    LC_COARSE_MAX_DIST  = float(os.environ.get("LC_COARSE_MAX_DIST",  str(_d["LC_COARSE_MAX_DIST"])))
    LC_COARSE_ITER      = int  (os.environ.get("LC_COARSE_ITER",      str(_d["LC_COARSE_ITER"])))
    LC_COARSE_REJECT    = float(os.environ.get("LC_COARSE_REJECT",    str(_d["LC_COARSE_REJECT"])))
    LC_COARSE_MAX_DELTA = float(os.environ.get("LC_COARSE_MAX_DELTA", str(_d["LC_COARSE_MAX_DELTA"])))

USE_VIZ   = os.environ.get("VIZ", "0") == "1"
VIZ_VOXEL = float(os.environ.get("VIZ_VOXEL", "0.5"))
MAX_CONSEC_REJECT = int(os.environ.get("MAX_CONSEC_REJECT", "5"))

print("[PARAM] ICP_VOXEL =",        ICP_VOXEL)
print("[PARAM] ICP_MAX_DIST =",     ICP_MAX_DIST)
print("[PARAM] ICP_REJECT_SCORE =", ICP_REJECT_SCORE)
print("[PARAM] ICP_MAX_ITER =",     ICP_MAX_ITER)
print("[PARAM] MAX_DELTA_TRANS =",  MAX_DELTA_TRANS)
print("[PARAM] KEYFRAME_DIST =",    KEYFRAME_DIST)
print("[PARAM] KEYFRAME_ANG =",     KEYFRAME_ANG)
print("[PARAM] SUBMAP_WINDOW =",    SUBMAP_WINDOW)
print("[PARAM] SUBMAP_VOXEL =",     SUBMAP_VOXEL)
print("[PARAM] ENABLE_LC =",        ENABLE_LC)
if ENABLE_LC:
    print("[PARAM] LC_DIST =",            LC_DIST)
    print("[PARAM] LC_SCORE =",           LC_SCORE)
    print("[PARAM] LC_MAX_CAND =",        LC_MAX_CAND)
    print("[PARAM] LC_MIN_SEP =",         LC_MIN_SEP)
    print("[PARAM] LC_MAX_CORRECTION =",  LC_MAX_CORRECTION)
    print("[PARAM] LC_MAX_YAW_DEG =",     LC_MAX_YAW_DEG)
    if IS_TRACK3:
        print("[PARAM] LC_MAX_ROT_DEG =",      LC_MAX_ROT_DEG)
        print("[PARAM] LC_COOLDOWN =",         LC_COOLDOWN)
        print("[PARAM] LC_COARSE_VOXEL =",     LC_COARSE_VOXEL)
        print("[PARAM] LC_COARSE_MAX_DIST =",  LC_COARSE_MAX_DIST)
        print("[PARAM] LC_COARSE_ITER =",      LC_COARSE_ITER)
        print("[PARAM] LC_COARSE_REJECT =",    LC_COARSE_REJECT)
        print("[PARAM] LC_COARSE_MAX_DELTA =", LC_COARSE_MAX_DELTA)
    else:
        print("[PARAM] LC_ICP_MAX_DIST =",    LC_ICP_MAX_DIST)
        print("[PARAM] LC_ICP_MAX_ITER =",    LC_ICP_MAX_ITER)
        print("[PARAM] LC_ICP_VOXEL =",       LC_ICP_VOXEL)
        print("[PARAM] LC_FITNESS_DIST =",    LC_FITNESS_DIST)
        print("[PARAM] LC_PGO_BATCH =",       LC_PGO_BATCH)
        print("[PARAM] LC_PGO_COOLDOWN =",    LC_PGO_COOLDOWN)
        print("[PARAM] LC_HIST_DEDUP =",      LC_HIST_DEDUP)


# ============================================================
# Helpers
# ============================================================

def check_rotation_matrix(T):
    Rm = T[:3, :3]
    if not np.isfinite(Rm).all():
        return False
    return abs(np.linalg.det(Rm) - 1.0) <= 0.1


def downsample_submap(pts, voxel_size):
    if pts is None or len(pts) == 0:
        return pts
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return pts
    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = np.ascontiguousarray(pts)
    if len(pts) == 0 or voxel_size <= 0:
        return pts
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(float(voxel_size))
        result = np.asarray(pcd.points, dtype=np.float64)
        return result if len(result) > 0 else pts
    except Exception as e:
        print(f"[WARN] downsample_submap failed ({e}).", flush=True)
        return pts


def yaw_from_R(Rm):
    return np.arctan2(Rm[1, 0], Rm[0, 0])


def distribute_pose_corrections(poses, kf_frame_indices, kf_opt_poses):
    new_poses = [p.copy() for p in poses]
    if len(kf_frame_indices) == 0 or len(kf_opt_poses) == 0:
        return new_poses
    usable = min(len(kf_frame_indices), len(kf_opt_poses))
    for k in range(usable - 1):
        frame_A = kf_frame_indices[k]
        frame_B = kf_frame_indices[k + 1]
        if frame_B <= frame_A:
            continue
        orig_A, orig_B = poses[frame_A], poses[frame_B]
        opt_A, opt_B   = kf_opt_poses[k], kf_opt_poses[k + 1]
        corr_A = opt_A @ np.linalg.inv(orig_A)
        corr_B = opt_B @ np.linalg.inv(orig_B)
        rot_A, rot_B   = ScipyR.from_matrix(corr_A[:3, :3]), ScipyR.from_matrix(corr_B[:3, :3])
        trans_A, trans_B = corr_A[:3, 3], corr_B[:3, 3]
        try:
            key_rots = ScipyR.from_quat(np.vstack([rot_A.as_quat(), rot_B.as_quat()]))
            slerp = Slerp([0, 1], key_rots)
        except Exception:
            slerp = None
        for fi in range(frame_A, frame_B):
            ratio = (fi - frame_A) / float(frame_B - frame_A)
            rot_t   = slerp(ratio).as_matrix() if slerp else rot_A.as_matrix()
            trans_t = trans_A + ratio * (trans_B - trans_A)
            corr_t = np.eye(4)
            corr_t[:3, :3] = rot_t
            corr_t[:3, 3]  = trans_t
            new_poses[fi]  = corr_t @ poses[fi]
    last_frame = kf_frame_indices[usable - 1]
    last_corr  = kf_opt_poses[usable - 1] @ np.linalg.inv(poses[last_frame])
    for fi in range(last_frame, len(poses)):
        new_poses[fi] = last_corr @ poses[fi]
    return new_poses


# ============================================================
# Loop closure
# ============================================================

class LoopClosure:
    def __init__(self, icp, dist_thresh, fitness_thresh, fitness_dist,
                 max_candidates, min_separation, max_loop_correction,
                 max_yaw_correction_deg):
        self.icp               = icp
        self.dist_thresh       = dist_thresh
        self.fitness_thresh    = fitness_thresh
        self.fitness_dist      = fitness_dist
        self.max_candidates    = max_candidates
        self.min_separation    = min_separation
        self.max_loop_correction = max_loop_correction
        self.max_yaw_correction  = np.deg2rad(max_yaw_correction_deg)
        self.keyframes = []
        self.poses     = []

    def add_keyframe(self, pts, pose):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        self.keyframes.append(pts.copy())
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    def update_poses(self, new_kf_poses):
        for i in range(min(len(new_kf_poses), len(self.poses))):
            self.poses[i] = np.asarray(new_kf_poses[i], dtype=np.float64).copy()

    def _find_candidates(self, curr_idx):
        if curr_idx <= self.min_separation:
            return []
        curr_xyz = self.poses[curr_idx][:3, 3]
        max_old  = curr_idx - self.min_separation
        info = []
        for idx in range(max_old):
            dist = np.linalg.norm(curr_xyz - self.poses[idx][:3, 3])
            if dist < self.dist_thresh:
                info.append((dist, idx))
        info.sort(key=lambda x: x[0])
        return [idx for _, idx in info[: self.max_candidates]]

    @staticmethod
    def _fitness_score(src_local, tgt_local, T_local, max_dist):
        """Symmetric inlier ratio: 0=perfect, 1=no match."""
        from scipy.spatial import cKDTree
        src_t = (T_local[:3, :3] @ src_local.T).T + T_local[:3, 3]

        tree_tgt  = cKDTree(tgt_local)
        d_fwd, _  = tree_tgt.query(src_t, workers=1)
        fwd_ratio = float(np.sum(d_fwd < max_dist)) / max(1, len(src_t))

        tree_src  = cKDTree(src_t)
        d_bwd, _  = tree_src.query(tgt_local, workers=1)
        bwd_ratio = float(np.sum(d_bwd < max_dist)) / max(1, len(tgt_local))

        return 1.0 - 0.5 * (fwd_ratio + bwd_ratio)

    def _lc_align(self, src_pts, tgt_pts, init_T=None):
        """Multi-scale GICP alignment for loop closure."""
        src = np.asarray(src_pts, dtype=np.float64)
        tgt = np.asarray(tgt_pts, dtype=np.float64)
        src = src[np.isfinite(src).all(axis=1)]
        tgt = tgt[np.isfinite(tgt).all(axis=1)]
        src = np.ascontiguousarray(src[:, :3])
        tgt = np.ascontiguousarray(tgt[:, :3])
        if len(src) < 50 or len(tgt) < 50:
            return np.eye(4), np.inf
        if init_T is None:
            init_T = np.eye(4, dtype=np.float64)
        init_T = np.ascontiguousarray(np.asarray(init_T, dtype=np.float64))

        # Stage 1: coarse alignment
        coarse_voxel = max(self.icp.voxel_size * 2.0, 1.0)
        coarse_corr  = max(self.icp.max_correspondence_distance * 1.5, 5.0)
        try:
            result1 = sg.align(
                target_points=tgt,
                source_points=src,
                init_T_target_source=init_T,
                registration_type="GICP",
                downsampling_resolution=coarse_voxel,
                max_correspondence_distance=coarse_corr,
                num_threads=self.icp.num_threads,
                max_iterations=self.icp.max_iterations,
                verbose=False,
            )
            T_coarse = np.asarray(result1.T_target_source, dtype=np.float64)
            if T_coarse.shape != (4, 4) or not np.isfinite(T_coarse).all():
                T_coarse = init_T.copy()
        except Exception:
            T_coarse = init_T.copy()

        # Stage 2: fine alignment
        try:
            result2 = sg.align(
                target_points=tgt,
                source_points=src,
                init_T_target_source=np.ascontiguousarray(T_coarse),
                registration_type="GICP",
                downsampling_resolution=self.icp.voxel_size,
                max_correspondence_distance=self.icp.max_correspondence_distance,
                num_threads=self.icp.num_threads,
                max_iterations=self.icp.max_iterations,
                verbose=False,
            )
            T_fine = np.asarray(result2.T_target_source, dtype=np.float64)
            if T_fine.shape != (4, 4) or not np.isfinite(T_fine).all():
                return T_coarse, float(getattr(result2, 'error', np.inf))
            return T_fine, float(result2.error)
        except Exception as e:
            print(f"[LC-ICP] fine align failed: {e}", flush=True)
            return T_coarse, np.inf

    def _match(self, curr_idx, candidates, submap_mgr):
        if not candidates:
            return None
        curr_pts       = self.keyframes[curr_idx]
        curr_pose_odom = self.poses[curr_idx]
        R_inv = curr_pose_odom[:3, :3].T
        t_inv = -R_inv @ curr_pose_odom[:3, 3]

        best_idx, best_T, best_score = None, None, np.inf

        for hist_idx in candidates:
            if hist_idx >= len(submap_mgr.submaps):
                continue
            hist_submap_world = submap_mgr.submaps[hist_idx]
            if hist_submap_world is None or len(hist_submap_world) < 50:
                continue
            hist_pose = self.poses[hist_idx]

            hist_submap_local = (R_inv @ hist_submap_world.T).T + t_inv
            if len(hist_submap_local) < 50:
                continue

            T_local, _ = self._lc_align(curr_pts, hist_submap_local)

            fitness = self._fitness_score(
                curr_pts, hist_submap_local, T_local, self.fitness_dist,
            )

            T_world_curr = curr_pose_odom @ T_local
            T_hist_curr  = np.linalg.inv(hist_pose) @ T_world_curr

            T_odom_rel  = np.linalg.inv(hist_pose) @ curr_pose_odom
            correction  = np.linalg.inv(T_odom_rel) @ T_hist_curr
            corr_trans  = np.linalg.norm(correction[:3, 3])
            corr_yaw    = abs(yaw_from_R(correction[:3, :3]))

            if corr_trans > self.max_loop_correction:
                continue
            if corr_yaw > self.max_yaw_correction:
                continue

            if fitness < best_score:
                best_score = fitness
                best_idx   = hist_idx
                best_T     = T_hist_curr

        if best_idx is None:
            return None

        if best_score < self.fitness_thresh:
            print(
                f"[LoopClosure] accepted: {best_idx} -> {curr_idx}, "
                f"fitness={best_score:.4f}",
                flush=True,
            )
            return best_idx, best_T, best_score

        print(
            f"[LoopClosure] rejected: curr={curr_idx}, "
            f"best_cand={best_idx}, fitness={best_score:.4f} "
            f"(thresh={self.fitness_thresh})",
            flush=True,
        )
        return None

    def detect(self, submap_mgr, cooldown_active=False, blocked_hist=None):
        curr_idx = len(self.keyframes) - 1
        if curr_idx < self.min_separation:
            return None
        if cooldown_active:
            return None
        candidates = self._find_candidates(curr_idx)
        if blocked_hist:
            candidates = [c for c in candidates if c not in blocked_hist]
        return self._match(curr_idx, candidates, submap_mgr)


# ============================================================
# Track3-specific implementations
# ============================================================

class ICPTrack3:
    """GICP wrapper with an optional delta-translation gate (disabled for
    loop closure, which is allowed to correct large accumulated drift)."""

    def __init__(self, voxel_size=0.25, max_correspondence_distance=2.5,
                 num_threads=4, reject_score=1.0, min_range=0.5, max_range=100.0,
                 min_z=-10.0, max_z=30.0, max_iterations=50, max_delta_trans=5.0):
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        self.num_threads = num_threads
        self.reject_score = reject_score
        self.min_range = min_range
        self.max_range = max_range
        self.min_z = min_z
        self.max_z = max_z
        self.max_iterations = max_iterations
        self.max_delta_trans = max_delta_trans

    def _filter_points(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        pts = pts[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return pts
        r = np.linalg.norm(pts, axis=1)
        mask = ((r > self.min_range) & (r < self.max_range)
                & (pts[:, 2] > self.min_z) & (pts[:, 2] < self.max_z))
        return np.ascontiguousarray(pts[mask], dtype=np.float64)

    def align(self, src_pts, tgt_pts, init_T=np.eye(4), check_delta=True):
        src = self._filter_points(src_pts)
        tgt = self._filter_points(tgt_pts)
        init_T = np.asarray(init_T, dtype=np.float64)
        if init_T.shape != (4, 4):
            init_T = np.eye(4, dtype=np.float64)
        init_T = np.ascontiguousarray(init_T, dtype=np.float64)
        if len(src) < 50 or len(tgt) < 50:
            return init_T.copy(), np.inf
        try:
            result = sg.align(
                target_points=tgt,
                source_points=src,
                init_T_target_source=init_T,
                registration_type="GICP",
                downsampling_resolution=self.voxel_size,
                max_correspondence_distance=self.max_correspondence_distance,
                num_threads=self.num_threads,
                max_iterations=self.max_iterations,
                verbose=False,
            )
            T = np.asarray(result.T_target_source, dtype=np.float64)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                return init_T.copy(), np.inf
            raw_error = float(result.error)
            num_inliers = int(getattr(result, "num_inliers", len(src)))
            score = raw_error / max(1, num_inliers)
            if not np.isfinite(score) or score > self.reject_score:
                return init_T.copy(), np.inf
            if check_delta:
                delta = np.linalg.inv(init_T) @ T
                if np.linalg.norm(delta[:3, 3]) > self.max_delta_trans:
                    return init_T.copy(), np.inf
            return T, score
        except Exception as e:
            print(f"[ICP] align failed: {e}", flush=True)
            return init_T.copy(), np.inf


class LoopClosureTrack3:
    """Conservative loop closure: coarse-then-fine GICP against historical
    world submaps, with translation / rotation / yaw correction gates."""

    def __init__(self, icp, icp_coarse=None, dist_thresh=8.0, fitness_thresh=0.06,
                 max_candidates=15, min_separation=80, max_loop_correction=3.0,
                 max_rot_correction_deg=20.0, max_yaw_correction_deg=25.0):
        self.icp = icp
        self.icp_coarse = icp_coarse
        self.dist_thresh = dist_thresh
        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_separation = min_separation
        self.max_loop_correction = max_loop_correction
        self.max_rot_correction = np.deg2rad(max_rot_correction_deg)
        self.max_yaw_correction = np.deg2rad(max_yaw_correction_deg)
        self.keyframes = []
        self.poses = []

    def add_keyframe(self, pts, pose):
        pts = np.asarray(pts)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        self.keyframes.append(pts)
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    def update_poses(self, new_poses):
        for i in range(min(len(new_poses), len(self.poses))):
            self.poses[i] = np.asarray(new_poses[i], dtype=np.float64).copy()

    @staticmethod
    def _yaw_from_R(Rm):
        return np.arctan2(Rm[1, 0], Rm[0, 0])

    @staticmethod
    def _rot_angle_from_R(Rm):
        return np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))

    def find_candidates(self, curr_idx):
        if curr_idx <= self.min_separation:
            return []
        curr_xyz = self.poses[curr_idx][:3, 3]
        info = []
        for idx in range(curr_idx - self.min_separation):
            dist = np.linalg.norm(curr_xyz - self.poses[idx][:3, 3])
            if dist < self.dist_thresh:
                info.append((dist, idx))
        info.sort(key=lambda x: x[0])
        return [idx for _, idx in info[: self.max_candidates]]

    def match(self, curr_idx, candidates, submap_mgr):
        if len(candidates) == 0:
            return None
        curr_pts = self.keyframes[curr_idx]
        curr_pose_odom = self.poses[curr_idx]

        best_idx = best_T_loop = best_corr_trans = best_corr_rot = None
        best_score = np.inf
        diag = None

        for hist_idx in candidates:
            if hist_idx >= len(submap_mgr.submaps):
                continue
            hist_pose = self.poses[hist_idx]
            hist_submap = submap_mgr.submaps[hist_idx]
            if hist_submap is None or len(hist_submap) < 50:
                continue

            if self.icp_coarse is not None:
                T_coarse, _ = self.icp_coarse.align(
                    src_pts=curr_pts, tgt_pts=hist_submap,
                    init_T=curr_pose_odom, check_delta=True,
                )
                if not np.isfinite(T_coarse).all():
                    T_coarse = curr_pose_odom
            else:
                T_coarse = curr_pose_odom

            T_world_curr, score = self.icp.align(
                src_pts=curr_pts, tgt_pts=hist_submap,
                init_T=T_coarse, check_delta=False,
            )
            if not np.isfinite(score):
                continue

            T_hist_curr = np.linalg.inv(hist_pose) @ T_world_curr
            T_odom_rel = np.linalg.inv(hist_pose) @ curr_pose_odom
            correction = np.linalg.inv(T_odom_rel) @ T_hist_curr
            corr_trans = float(np.linalg.norm(correction[:3, 3]))
            corr_rot = float(self._rot_angle_from_R(correction[:3, :3]))
            corr_yaw = float(abs(self._yaw_from_R(correction[:3, :3])))

            if diag is None or score < diag[1]:
                diag = (hist_idx, score, corr_trans, corr_rot, corr_yaw)

            if corr_trans > self.max_loop_correction:
                continue
            if corr_rot > self.max_rot_correction:
                continue
            if corr_yaw > self.max_yaw_correction:
                continue

            if score < best_score:
                best_score = score
                best_idx = hist_idx
                best_T_loop = T_hist_curr
                best_corr_trans = corr_trans
                best_corr_rot = corr_rot

        if best_idx is not None and best_score < self.fitness_thresh:
            print(
                f"[LoopClosure] accepted: {best_idx} -> {curr_idx}, "
                f"score={best_score:.4f}, corr_trans={best_corr_trans:.2f}m, "
                f"corr_rot={np.rad2deg(best_corr_rot):.1f}deg",
                flush=True,
            )
            return best_idx, best_T_loop, best_score

        if diag is not None:
            print(
                f"[LoopClosure] rejected at kf={curr_idx}: best_cand={diag[0]}, "
                f"score={diag[1]:.4f} (thr={self.fitness_thresh}), "
                f"corr_trans={diag[2]:.2f}m (thr={self.max_loop_correction}), "
                f"corr_rot={np.rad2deg(diag[3]):.1f}deg "
                f"(thr={np.rad2deg(self.max_rot_correction):.0f}), "
                f"corr_yaw={np.rad2deg(diag[4]):.1f}deg "
                f"(thr={np.rad2deg(self.max_yaw_correction):.0f})",
                flush=True,
            )
        return None

    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        if curr_idx < self.min_separation:
            return None
        candidates = self.find_candidates(curr_idx)
        if curr_idx % 20 == 0:
            curr_xyz = self.poses[curr_idx][:3, 3]
            max_old_idx = curr_idx - self.min_separation
            if max_old_idx > 0:
                min_dist = min(np.linalg.norm(curr_xyz - self.poses[i][:3, 3])
                               for i in range(max_old_idx))
                print(f"[LC_DEBUG] kf={curr_idx}, candidates={len(candidates)}, "
                      f"nearest_hist_dist={min_dist:.2f}", flush=True)
        if len(candidates) == 0:
            return None
        return self.match(curr_idx, candidates, submap_mgr)


def distribute_pose_corrections_track3(poses, kf_frame_indices, kf_opt_poses):
    """Rigid block correction: each keyframe's correction is applied to all
    frames up to the next keyframe (no interpolation)."""
    new_poses = [p.copy() for p in poses]
    usable = min(len(kf_frame_indices), len(kf_opt_poses))
    if usable == 0:
        return new_poses
    for k in range(usable):
        frame_kf = kf_frame_indices[k]
        T_correct = kf_opt_poses[k] @ np.linalg.inv(poses[frame_kf])
        next_kf = kf_frame_indices[k + 1] if k + 1 < usable else len(poses)
        for fi in range(frame_kf, next_kf):
            new_poses[fi] = T_correct @ poses[fi]
    return new_poses


# ============================================================
# Initialization
# ============================================================

if IS_TRACK3:
    icp = ICPTrack3(
        voxel_size=ICP_VOXEL,
        max_correspondence_distance=ICP_MAX_DIST,
        num_threads=4,
        reject_score=ICP_REJECT_SCORE,
        min_range=0.5,
        max_range=100.0,
        min_z=-10.0,
        max_z=30.0,
        max_iterations=ICP_MAX_ITER,
        max_delta_trans=MAX_DELTA_TRANS,
    )
    lc_icp = None
else:
    icp = ICP(
        voxel_size=ICP_VOXEL,
        max_correspondence_distance=ICP_MAX_DIST,
        num_threads=4,
        reject_score=ICP_REJECT_SCORE,
        min_range=0.5,
        max_range=100.0,
        min_z=-10.0,
        max_z=30.0,
        max_iterations=ICP_MAX_ITER,
        max_delta_trans=MAX_DELTA_TRANS,
    )
    lc_icp = ICP(
        voxel_size=LC_ICP_VOXEL,
        max_correspondence_distance=LC_ICP_MAX_DIST,
        num_threads=4,
        reject_score=np.inf,
        min_range=0.0,
        max_range=9999.0,
        min_z=-9999.0,
        max_z=9999.0,
        max_iterations=LC_ICP_MAX_ITER,
        max_delta_trans=np.inf,
    ) if ENABLE_LC else None

submap_mgr = SubmapManager(window_size=SUBMAP_WINDOW)
kf_manager = KeyframeManager(dist_thresh=KEYFRAME_DIST, ang_thresh=KEYFRAME_ANG)

pg = None
lc = None
if ENABLE_LC and IS_TRACK3:
    pg = PoseGraph(mode="batch")
    pg.add_prior()
    lc_icp_coarse = ICPTrack3(
        voxel_size=LC_COARSE_VOXEL,
        max_correspondence_distance=LC_COARSE_MAX_DIST,
        num_threads=4,
        reject_score=LC_COARSE_REJECT,
        min_range=0.5,
        max_range=100.0,
        min_z=-10.0,
        max_z=30.0,
        max_iterations=LC_COARSE_ITER,
        max_delta_trans=LC_COARSE_MAX_DELTA,
    )
    lc = LoopClosureTrack3(
        icp=icp,
        icp_coarse=lc_icp_coarse,
        dist_thresh=LC_DIST,
        fitness_thresh=LC_SCORE,
        max_candidates=LC_MAX_CAND,
        min_separation=LC_MIN_SEP,
        max_loop_correction=LC_MAX_CORRECTION,
        max_rot_correction_deg=LC_MAX_ROT_DEG,
        max_yaw_correction_deg=LC_MAX_YAW_DEG,
    )
elif ENABLE_LC:
    pg = PoseGraph()
    pg.add_prior()
    pg.add_node(0, np.eye(4))
    lc = LoopClosure(
        icp=lc_icp,
        dist_thresh=LC_DIST,
        fitness_thresh=LC_SCORE,
        fitness_dist=LC_FITNESS_DIST,
        max_candidates=LC_MAX_CAND,
        min_separation=LC_MIN_SEP,
        max_loop_correction=LC_MAX_CORRECTION,
        max_yaw_correction_deg=LC_MAX_YAW_DEG,
    )

viz = Visualizer() if USE_VIZ else None

prev_pose      = np.eye(4)
last_rel_pose  = np.eye(4)
consec_rejects = 0

poses      = []
timestamps = []

pose_idx = -1

kf_frame_indices = []
kf_poses         = []
last_kf_pose     = np.eye(4)
current_kf_idx   = 0

full_map     = np.zeros((0, 3))
kf_poses_viz = []
per_frame_deltas = []
lc_count = 0
lc_pending      = 0
lc_cooldown_rem = 0
lc_hist_recent  = {}
last_loop_kf    = -10 ** 9   # Track3: keyframe index of last accepted loop


# ============================================================
# Main loop
# ============================================================

for frame_idx, f in enumerate(files):
    if frame_idx == 0 or (frame_idx + 1) % 50 == 0:
        print(f"[INFO] Processing frame {frame_idx + 1}/{len(files)}", flush=True)

    pts = load_pcd(f)
    if pts is None or len(pts) == 0:
        print(f"[WARN] Empty point cloud: {f}", flush=True)
        continue

    if pts.shape[1] > 3:
        pts = pts[:, :3]

    timestamp = os.path.basename(f).replace(".pcd", "")
    pose_idx += 1

    if pose_idx == 0:
        curr_pose = np.eye(4)
        poses.append(curr_pose.copy())
        timestamps.append(timestamp)

        submap_mgr.add_keyframe(pts, curr_pose)
        kf_frame_indices.append(pose_idx)
        kf_poses.append(curr_pose.copy())

        if ENABLE_LC:
            lc.add_keyframe(pts, curr_pose)

        prev_pose    = curr_pose.copy()
        last_kf_pose = curr_pose.copy()

        if USE_VIZ:
            kf_poses_viz.append(curr_pose.copy())
            full_map = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]

        print("[INFO] Initial frame added.", flush=True)
        continue

    init_T = prev_pose @ last_rel_pose

    if IS_TRACK3:
        submap_pts = submap_mgr.get_latest_submap()
        if submap_pts is None or len(submap_pts) < 50:
            curr_pose = init_T.copy()
            print("[WARN] No valid submap, use predicted pose.", flush=True)
        else:
            T, score = icp.align(src_pts=pts, tgt_pts=submap_pts, init_T=init_T, check_delta=True)
            if np.isfinite(score) and check_rotation_matrix(T):
                curr_pose     = T.copy()
                last_rel_pose = np.linalg.inv(prev_pose) @ curr_pose
            else:
                curr_pose = init_T.copy()
                print("[WARN] ICP rejected or failed, use predicted pose.", flush=True)
    else:
        submap_pts_world = submap_mgr.get_latest_submap()
        if submap_pts_world is not None and len(submap_pts_world) >= 50:
            submap_pts_world = downsample_submap(submap_pts_world, SUBMAP_VOXEL)

        if submap_pts_world is not None and len(submap_pts_world) >= 50:
            R_inv = init_T[:3, :3].T
            t_inv = -R_inv @ init_T[:3, 3]
            submap_pts_local = (R_inv @ submap_pts_world.T).T + t_inv
        else:
            submap_pts_local = None

        if submap_pts_local is None or len(submap_pts_local) < 50:
            curr_pose = init_T.copy()
            consec_rejects += 1
        else:
            T_local, score = icp.align(pts, submap_pts_local, np.eye(4))
            T = init_T @ T_local

            if np.isfinite(score) and check_rotation_matrix(T):
                curr_pose      = T.copy()
                last_rel_pose  = np.linalg.inv(prev_pose) @ curr_pose
                consec_rejects = 0
            else:
                curr_pose = init_T.copy()
                consec_rejects += 1
                if consec_rejects >= MAX_CONSEC_REJECT:
                    last_rel_pose  = np.eye(4)
                    consec_rejects = 0

    poses.append(curr_pose.copy())
    timestamps.append(timestamp)

    step = float(np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3]))
    per_frame_deltas.append(step)

    if kf_manager.is_keyframe(curr_pose):
        current_kf_idx += 1

        submap_mgr.add_keyframe(pts, curr_pose)
        kf_frame_indices.append(pose_idx)
        kf_poses.append(curr_pose.copy())

        rel_pose  = np.linalg.inv(last_kf_pose) @ curr_pose

        if ENABLE_LC and IS_TRACK3:
            pg.add_node(current_kf_idx, curr_pose)
            pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose)
            lc.add_keyframe(pts, curr_pose)

            if current_kf_idx - last_loop_kf >= LC_COOLDOWN:
                loop_result = lc.detect(submap_mgr)
            else:
                loop_result = None

            if loop_result is not None:
                loop_idx, T_loop, loop_score = loop_result
                lc_count += 1
                pg.add_loop(loop_idx, current_kf_idx, T_loop, robust=True)

                kf_opt_dict  = pg.optimize()
                sorted_keys  = sorted(kf_opt_dict.keys())
                kf_opt_poses = [kf_opt_dict[k] for k in sorted_keys]

                submap_mgr.update_poses(kf_opt_poses)
                lc.update_poses(kf_opt_poses)
                poses = distribute_pose_corrections_track3(poses, kf_frame_indices, kf_opt_poses)
                curr_pose = kf_opt_poses[-1].copy()
                poses[-1] = curr_pose.copy()
                kf_poses  = kf_opt_poses
                last_loop_kf = current_kf_idx
                # NOTE: keep the pre-PGO velocity estimate; poses[-2] and
                # poses[-1] may belong to different correction blocks.
                print(
                    f"[PGO] optimized at keyframe {current_kf_idx}, "
                    f"loop={loop_idx}->{current_kf_idx}, score={loop_score:.6f}",
                    flush=True,
                )

        elif ENABLE_LC:
            pg.add_node(current_kf_idx, curr_pose)
            pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose)
            lc.add_keyframe(pts, curr_pose)

            if lc_cooldown_rem > 0:
                lc_cooldown_rem -= 1

            blocked = {h for h, last_kf in lc_hist_recent.items()
                       if current_kf_idx - last_kf < LC_HIST_DEDUP}

            loop_result = lc.detect(
                submap_mgr,
                cooldown_active=(lc_cooldown_rem > 0),
                blocked_hist=blocked,
            )

            if loop_result is not None:
                loop_idx, T_loop, loop_score = loop_result
                lc_count   += 1
                lc_pending += 1
                pg.add_loop(loop_idx, current_kf_idx, T_loop)
                lc_hist_recent[loop_idx] = current_kf_idx
                print(f"[LC] queued loop {loop_idx}->{current_kf_idx}, fitness={loop_score:.4f}, pending={lc_pending}/{LC_PGO_BATCH}", flush=True)

            if lc_pending >= LC_PGO_BATCH:
                lc_pending = 0
                kf_opt_dict  = pg.optimize()
                sorted_keys  = sorted(kf_opt_dict.keys())
                kf_opt_poses = [kf_opt_dict[k] for k in sorted_keys]

                if len(sorted_keys) == len(kf_frame_indices):
                    submap_mgr.update_poses(kf_opt_poses)
                    lc.update_poses(kf_opt_poses)
                    poses = distribute_pose_corrections(poses, kf_frame_indices, kf_opt_poses)
                    curr_pose = kf_opt_poses[-1].copy()
                    poses[-1] = curr_pose.copy()
                    kf_poses  = kf_opt_poses
                    if len(poses) >= 2:
                        last_rel_pose = np.linalg.inv(poses[-2]) @ poses[-1]
                    lc_cooldown_rem = LC_PGO_COOLDOWN
                    print(f"[PGO] optimized with {lc_count} total loops, cooldown={LC_PGO_COOLDOWN} kf", flush=True)

        last_kf_pose = curr_pose.copy()

        if USE_VIZ:
            kf_poses_viz.append(curr_pose.copy())
            pts_w    = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
            full_map = np.concatenate([full_map, pts_w], axis=0)
            try:
                fm_clean = np.ascontiguousarray(full_map[np.isfinite(full_map).all(axis=1)], dtype=np.float64)
                pcd_tmp        = o3d.geometry.PointCloud()
                pcd_tmp.points = o3d.utility.Vector3dVector(fm_clean)
                pcd_tmp        = pcd_tmp.voxel_down_sample(voxel_size=float(VIZ_VOXEL))
                full_map       = np.asarray(pcd_tmp.points, dtype=np.float64)
            except Exception:
                pass

    prev_pose = curr_pose.copy()

    if USE_VIZ:
        pts_curr_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
        viz.update(pts_curr_w, full_map, poses, kf_poses_viz)


# ============================================================
# Final stats
# ============================================================

if hasattr(icp, "success_count"):
    total_icp = (icp.success_count + icp.reject_score_count + icp.reject_delta_trans_count + icp.fail_count)
    icp_rate = icp.success_count / max(1, total_icp)
else:
    icp_rate = None
start_t  = poses[0][:3, 3]
end_t    = poses[-1][:3, 3]
end_dist = float(np.linalg.norm(end_t - start_t))
cum_path = sum(per_frame_deltas)
drift_rate = end_dist / max(1.0, cum_path)

all_xyz    = np.array([p[:3, 3] for p in poses])
n          = len(per_frame_deltas)
seg_labels = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
seg_jerks  = []
for qi in range(4):
    a = qi * n // 4
    b = (qi + 1) * n // 4
    seg = per_frame_deltas[a:b]
    if len(seg) >= 2:
        accs = [abs(seg[i] - seg[i-1]) for i in range(1, len(seg))]
        seg_jerks.append(float(np.sqrt(np.mean(np.array(accs) ** 2))))
    else:
        seg_jerks.append(0.0)

rms_jerk_global = float(np.sqrt(np.mean(np.array([abs(per_frame_deltas[i] - per_frame_deltas[i-1]) for i in range(1, n)]) ** 2))) if n >= 3 else 0.0

print()
print("=" * 60)
print("[FINAL STATS]")
print(f"  Target:                     {target}")
print(f"  Total frames:               {len(poses)}")
print(f"  Keyframes:                  {current_kf_idx + 1}")
print(f"  Loop closures applied:      {lc_count}")
if icp_rate is not None:
    print(f"  ICP success rate:           {icp_rate * 100:.1f}%")
if ENABLE_LC and lc_icp is not None:
    lc_total = (lc_icp.success_count + lc_icp.reject_score_count + lc_icp.reject_delta_trans_count + lc_icp.fail_count)
    lc_rate = lc_icp.success_count / max(1, lc_total)
    print(f"  LC ICP success rate:        {lc_rate * 100:.1f}%")

print()
print("── Proxy metrics ──────────────────────────────────────")
print(f"  Drift rate  (end/path):     {drift_rate:.4f}  "
      f"{'✓ good' if drift_rate < 0.02 else '△ moderate' if drift_rate < 0.05 else '✗ poor'}")
print(f"  Closure gap (start→end):    {end_dist:.3f} m")
print(f"  RMS jerk    (global):       {rms_jerk_global:.4f} m/f²")
for label, jerk in zip(seg_labels, seg_jerks):
    print(f"    {label}:  {jerk:.4f} m/f²")
print("=" * 60)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
rows = [[str(t)] + T.reshape(-1).tolist() for t, T in zip(timestamps, poses)]
df = pd.DataFrame(rows, columns=header.split(","))
df.to_csv(output_csv, index=False)
