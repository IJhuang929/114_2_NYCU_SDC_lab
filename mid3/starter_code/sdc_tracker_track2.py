from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


MAX_TRAJ = 5
class KalmanFilter:
    def __init__(self, dt=0.1):
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.P = np.eye(4) * 500.0
        self.Q = np.eye(4) * 0.5
        self.R = np.eye(2) * 0.5
        self.x = np.zeros((4, 1), dtype=float)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.flatten()[:2]

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(2, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

        return self.x.flatten()[:2]


class Tracker:
    def __init__(
        self,
        max_age: int = 8,
        min_hits: int = 1,
        max_distance: float = 5.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.next_track_id = 0
        self.tracks = []

    def update(self, detections: np.ndarray, ego_delta=None) -> np.ndarray:
        """
        Update the tracker with new detections.

        Expected detection format:
            [x, y, z, cluster_id, range_rate, rcs]

        Returned confirmed track format:
            [x, y, z, cluster_id, track_id]

        Note:
            This starter code does not implement the data association method.
            Students must complete match_tracks() or replace this tracker with
            their own implementation.
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6), dtype=float)
        if ego_delta is not None:
            dx, dy = ego_delta

            for track in self.tracks:
                track["mean"][0] -= dx
                track["mean"][1] -= dy

                if "kf" in track:
                    track["kf"].x[0, 0] -= dx
                    track["kf"].x[1, 0] -= dy

        matched, unmatched_detections, unmatched_tracks = self.match_tracks(detections)
        for track in self.tracks:
            track["matched_now"] = False

        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]

            updated_position = track["kf"].update(detection[:2])
            track["mean"] = updated_position
            track["range_rate"] = float(detection[4])
            track["rcs"] = float(detection[5])
            track["hits"] += 1
            track["age"] = 0
            track["state"] = "confirmed" if track["hits"] >= self.min_hits else "tentative"
            track["det"] = detection
            track["matched_now"] = True
            track["prev_states"].append(updated_position)

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]

            kf = KalmanFilter()
            kf.x[:2] = detection[:2].reshape(2, 1)
            self.tracks.append(
                {
                    "mean": detection[:2],
                    "kf": kf,
                    "range_rate": float(detection[4]),
                    "rcs": float(detection[5]),
                    "track_id": self.next_track_id,
                    "hits": 1,
                    "age": 0,
                    "state": "confirmed" if self.min_hits <= 1 else "tentative",
                    "det": detection,
                    "matched_now": True,
                    "prev_states": deque(maxlen=MAX_TRAJ),
                }
            )
            self.next_track_id += 1

        for track_idx in unmatched_tracks:
            if track_idx >= len(self.tracks):
                continue

            track = self.tracks[track_idx]
            track["age"] += 1
            if "kf" in track:
                track["mean"] = track.get("predicted_mean", track["mean"])

            if track["age"] > self.max_age:
                track["state"] = "deleted"

        self.tracks = [t for t in self.tracks if t["state"] != "deleted"]

        confirmed_tracks = []

        for track in self.tracks:
            if track["state"] != "confirmed":
                continue
            # 只輸出這一幀真的有配到 detection 的 track
            # 避免未配對 track 帶著上一幀的 cluster_id 輸出
            if not track.get("matched_now", False):
                continue

            det = track["det"]
            confirmed_tracks.append(
                [
                    float(track["mean"][0]),
                    float(track["mean"][1]),
                    float(det[2]),
                    int(det[3]),
                    int(track["track_id"]),
                ]
            )

        return np.asarray(confirmed_tracks, dtype=float)

    def match_tracks(self, detections: np.ndarray):
        """
        Associate current-frame detections with existing tracks.

        Required return values:
            matched:
                list of (track_index, detection_index)

            unmatched_detections:
                list of detection indices that are not assigned to any existing track

            unmatched_tracks:
                list of track indices that are not assigned to any current detection

        TODO-6:
            Implement your own data association strategy here.

        Important:
            This function currently returns no matches. Therefore, the starter
            code can run and produce a correctly formatted result.csv, but it
            will not produce meaningful tracking results until this TODO is
            completed.
        """
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(self.tracks) == 0 or len(detections) == 0:
            return matched, unmatched_detections, unmatched_tracks

        # 1. Kalman Filter prediction
        predicted_positions = []

        for track in self.tracks:
            if "kf" in track:
                pred = track["kf"].predict()
            else:
                pred = track["mean"]
            track["predicted_mean"] = pred
            predicted_positions.append(pred)

        track_xy = np.asarray(predicted_positions, dtype=float)
        det_xy = detections[:, :2].astype(float)

        diff = track_xy[:, None, :] - det_xy[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # 2. RangeRate cost
        RR_WEIGHT = 1.0
        track_rr = np.array(
            [track.get("range_rate", 0.0) for track in self.tracks],
            dtype=float,
        )
        det_rr = detections[:, 4].astype(float)
        rr_diff = np.abs(track_rr[:, None] - det_rr[None, :])

        # 3. RCS cost：先設 0，等於暫時不用 RCS
        RCS_WEIGHT = 0.0
        track_rcs = np.array(
            [track.get("rcs", 0.0) for track in self.tracks],
            dtype=float,
        )
        det_rcs = detections[:, 5].astype(float)
        rcs_diff = np.abs(track_rcs[:, None] - det_rcs[None, :])

        cost = dist + RR_WEIGHT * rr_diff + RCS_WEIGHT * rcs_diff

        RR_HARD_GATE = 3.0

        def solve(track_idxs, det_idxs, gate):
            out = []

            if not track_idxs or not det_idxs:
                return out

            sub_cost = cost[np.ix_(track_idxs, det_idxs)].copy()
            sub_dist = dist[np.ix_(track_idxs, det_idxs)]
            sub_rr   = rr_diff[np.ix_(track_idxs, det_idxs)]

            sub_cost[sub_dist > gate] = 1e6
            sub_cost[sub_rr > RR_HARD_GATE] = 1e6

            row_indices, col_indices = linear_sum_assignment(sub_cost)

            for r, c in zip(row_indices, col_indices):
                if sub_dist[r, c] > gate:
                    continue

                out.append((track_idxs[r], det_idxs[c]))

            return out

        confirmed_idx = [
            i for i, track in enumerate(self.tracks)
            if track["state"] == "confirmed"
        ]

        tentative_idx = [
            i for i, track in enumerate(self.tracks)
            if track["state"] != "confirmed"
        ]

        all_dets = list(range(len(detections)))

        matched_tracks = set()
        matched_detections = set()

        # Stage 1: confirmed tracks 優先配對
        for track_idx, detection_idx in solve(confirmed_idx, all_dets, self.max_distance):
            matched.append((track_idx, detection_idx))
            matched_tracks.add(track_idx)
            matched_detections.add(detection_idx)
            
        # # Stage 1.5: re-ID for unmatched confirmed tracks
        # # 目的：confirmed track 短暫沒配到時，用稍微寬一點的 gate 接回來，避免開新 ID
        # unmatched_confirmed_idx = [
        #     track_idx for track_idx in confirmed_idx
        #     if track_idx not in matched_tracks
        # ]

        # remaining_detections_for_reid = [
        #     detection_idx for detection_idx in all_dets
        #     if detection_idx not in matched_detections
        # ]

        # REID_GATE = self.max_distance * 1.4

        # for track_idx, detection_idx in solve(
        #     unmatched_confirmed_idx,
        #     remaining_detections_for_reid,
        #     REID_GATE
        # ):
        #     # re-ID 要更保守：range_rate 差太多就不要接
        #     rr_err = abs(
        #         self.tracks[track_idx].get("range_rate", 0.0)
        #         - float(detections[detection_idx, 4])
        #     )

        #     if rr_err > 2.0:
        #         continue

        #     matched.append((track_idx, detection_idx))
        #     matched_tracks.add(track_idx)
        #     matched_detections.add(detection_idx)

        # Stage 2: tentative tracks 再配剩下的 detections
        remaining_detections = [
            detection_idx for detection_idx in all_dets
            if detection_idx not in matched_detections
        ]

        for track_idx, detection_idx in solve(tentative_idx, remaining_detections, self.max_distance):
            matched.append((track_idx, detection_idx))
            matched_tracks.add(track_idx)
            matched_detections.add(detection_idx)

        unmatched_tracks = [
            track_idx for track_idx in range(len(self.tracks))
            if track_idx not in matched_tracks
        ]

        unmatched_detections = [
            detection_idx for detection_idx in all_dets
            if detection_idx not in matched_detections
        ]
        return matched, unmatched_detections, unmatched_tracks
