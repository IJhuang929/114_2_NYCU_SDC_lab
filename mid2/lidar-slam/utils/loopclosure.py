import numpy as np

'''
Make sure you set the parameters.
'''
class LoopClosure:
    def __init__(
        self,
        icp,
        icp_coarse=None,
        dist_thresh=8.0,
        fitness_thresh=0.06,
        max_candidates=15,
        min_separation=80,
        max_loop_correction=3.0,
        max_rot_correction_deg=20.0,
        max_yaw_correction_deg=25.0,
    ):
        """
        icp: ICP instance (small_gicp wrapper)
        """

        self.icp = icp
        self.icp_coarse = icp_coarse
        # ===== parameter =====
        self.dist_thresh = dist_thresh

        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_separation = min_separation
        self.max_loop_correction = max_loop_correction
        self.max_rot_correction = np.deg2rad(max_rot_correction_deg)
        self.max_yaw_correction = np.deg2rad(max_yaw_correction_deg)
        # ===== state =====
        self.keyframes = []
        self.poses = []
        
        self.last_loop_accum = 0.0

    # =========================
    #       Add Keyframe
    # =========================
    def add_keyframe(self, pts, pose):
        pts = np.asarray(pts)
        
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        # local frame points
        self.keyframes.append(pts)
        
        # world frame poses
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    def update_poses(self, new_poses):
        for i in range(min(len(new_poses), len(self.poses))):
            self.poses[i] = np.asarray(new_poses[i], dtype=np.float64).copy()

    def _yaw_from_R(self, Rm):
        return np.arctan2(Rm[1, 0], Rm[0, 0])

    def _rot_angle_from_R(self, Rm):
        return np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))
    
    # ==========================
    #    Candidate selection
    # ==========================
    def find_candidates(self, curr_idx):
        """
        [TODO] 
        
            Choose which candidates to verify loop closure,
            You could base  on distance, time interval 
            
            return: index of candidates
        """
        
        candidates = []
        if curr_idx <= self.min_separation:
            return candidates

        curr_pose = np.asarray(self.poses[curr_idx], dtype=np.float64)
        curr_xyz = curr_pose[:3, 3]

        candidate_info = []

        max_old_idx = curr_idx - self.min_separation

        for idx in range(max_old_idx):
            old_pose = np.asarray(self.poses[idx], dtype=np.float64)
            old_xyz = old_pose[:3, 3]

            dist = np.linalg.norm(curr_xyz - old_xyz)

            if dist < self.dist_thresh:
                candidate_info.append((dist, idx))

        candidate_info.sort(key=lambda x: x[0])
        candidates = [idx for _, idx in candidate_info[: self.max_candidates]]


        return candidates

    # =========================
    #       ICP matching
    # =========================
    def match(self, curr_idx, candidates, submap_mgr):
        '''
        curr_idx: current
        candidates: candidates that you choose to do loop closure detection with current frame
        submap_mgr: use as submap_mgr.submaps[idx] to get idx-th submap points(those are in world frame)
        '''
        if len(candidates) == 0:
            return None


        curr_pts = self.keyframes[curr_idx]
        curr_pose_odom = self.poses[curr_idx]

        best_idx = None
        best_T_loop = None
        best_score = np.inf
        best_corr_trans = None
        best_corr_rot = None

        # Diagnostics for the lowest-score candidate that passed ICP but
        # failed a gate, so thresholds can be tuned from the log.
        diag = None


        for hist_idx in candidates:
            """
            TODO: 
            1. Perform scan matching between curr_idx and candidate idx.
            2. Evaluate the match quality.
            3. Update best_score and best_T
            
            4. If valid loop is found, return corresponding results,
               otherwise return None.
            """
            if hist_idx >= len(submap_mgr.submaps):
                continue

            hist_pose = self.poses[hist_idx]
            hist_submap = submap_mgr.submaps[hist_idx]

            if hist_submap is None or len(hist_submap) < 50:
                continue

            # Coarse alignment first: bridges large odometry drift that the
            # fine ICP's small correspondence distance cannot. Guarded
            # (reject_score / check_delta on the coarse ICP) so a divergent
            # or flipped coarse solution falls back to the odometry init
            # instead of poisoning the fine stage.
            if self.icp_coarse is not None:
                T_coarse, _ = self.icp_coarse.align(
                    src_pts=curr_pts,
                    tgt_pts=hist_submap,
                    init_T=curr_pose_odom,
                    check_delta=True,
                )
                if not np.isfinite(T_coarse).all():
                    T_coarse = curr_pose_odom
            else:
                T_coarse = curr_pose_odom

            # Align current scan to historical world submap.
            # check_delta=False because loop closure may correct accumulated drift.
            T_world_curr, score = self.icp.align(
                src_pts=curr_pts,
                tgt_pts=hist_submap,
                init_T=T_coarse,
                check_delta=False,
            )

            if not np.isfinite(score):
                continue

            # Pose graph edge: historical keyframe -> current keyframe
            T_hist_curr = np.linalg.inv(hist_pose) @ T_world_curr

            # Reject suspicious constraints by comparing against odometry.
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
            

    
    # =========================
    #            API
    # =========================
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        # Could block some too-early detections
        if curr_idx < self.min_separation:
            return None

        candidates = self.find_candidates(curr_idx)
        #debug
        if curr_idx % 20 == 0:
            curr_xyz = self.poses[curr_idx][:3, 3]
            # 找最近的歷史 keyframe 距離
            max_old_idx = curr_idx - self.min_separation
            if max_old_idx > 0:
                min_dist = np.inf
                for idx in range(max_old_idx):
                    d = np.linalg.norm(curr_xyz - self.poses[idx][:3, 3])
                    if d < min_dist:
                        min_dist = d
                print(f"[LC_DEBUG] kf={curr_idx}, candidates={len(candidates)}, "
                    f"nearest_hist_dist={min_dist:.2f}", flush=True)

        if len(candidates) == 0:
            return None

        return self.match(curr_idx, candidates, submap_mgr)

