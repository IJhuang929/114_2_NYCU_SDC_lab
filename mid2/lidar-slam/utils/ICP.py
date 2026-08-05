import os
import small_gicp as sg
import numpy as np


_TRACK_DEFAULTS = {
    "Track1": {
        "voxel_size": 0.24,
        "max_correspondence_distance": 3.0,
        "max_iterations": 60,
        "max_delta_trans": 10.0,
    },
    "Track2": {
        "voxel_size": 0.25,
        "max_correspondence_distance": 2.5,
        "max_iterations": 50,
        "max_delta_trans": 5.0,
    },
    "Track3": {
        "voxel_size": 0.1,
        "max_correspondence_distance": 2.5,
        "max_iterations": 80,
        "max_delta_trans": 10.0, 
    },
}


class ICP:
    def __init__(
        self,
        voxel_size=None,
        max_correspondence_distance=None,
        num_threads=4,
        reject_score=1.0,
        min_range=0.5,
        max_range=100.0,
        min_z=-10.0,
        max_z=30.0,
        max_iterations=None,
        max_delta_trans=None,
    ):
        target = os.environ.get("TARGET", "Track2")
        defaults = _TRACK_DEFAULTS.get(target, _TRACK_DEFAULTS["Track2"])

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

        self.success_count = 0
        self.reject_score_count = 0
        self.reject_delta_trans_count = 0
        self.fail_count = 0

    def _filter_points(self, pts):
        pts = np.asarray(pts, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)

        pts = pts[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]

        if len(pts) == 0:
            return pts

        r = np.linalg.norm(pts, axis=1)

        mask = (
            (r > self.min_range)
            & (r < self.max_range)
            & (pts[:, 2] > self.min_z)
            & (pts[:, 2] < self.max_z)
        )

        pts = pts[mask]
        return np.ascontiguousarray(pts, dtype=np.float64)

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

            if T.shape != (4, 4):
                return init_T.copy(), np.inf

            if not np.isfinite(T).all():
                return init_T.copy(), np.inf

            raw_error = float(result.error)
            num_inliers = int(getattr(result, "num_inliers", len(src)))
            score = raw_error / max(1, num_inliers)

            if not np.isfinite(score):
                return init_T.copy(), np.inf

            if score > self.reject_score:
                return init_T.copy(), np.inf

            if check_delta:
                delta = np.linalg.inv(init_T) @ T
                delta_trans = np.linalg.norm(delta[:3, 3])

                if delta_trans > self.max_delta_trans:
                    return init_T.copy(), np.inf

            return T, score

        except Exception as e:
            print(f"[ICP] align failed: {e}", flush=True)
            return init_T.copy(), np.inf
