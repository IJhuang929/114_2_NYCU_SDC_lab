import numpy as np
from math import cos, sin, atan2

def wrap_angle(a):
    return atan2(sin(a), cos(a))

class ExtendedKalmanFilter:
    def __init__(self, x, y, yaw):
        # State: [x, y, yaw]
        self.pose = np.array([x, y, yaw], dtype=np.float64)

        # State covariance
        self.S = np.identity(3) * 1.0

        # GPS observes only (x, y) -> H is 2x3
        self.C = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])

        # State transition noise (overwritten from odom covariance each step)
        self.R = np.identity(3) * 1.0

        # Measurement noise (overwritten from gps covariance each step)
        self.Q = np.identity(2) * 1.0
        print("Initialize Kalman Filter")

    def predict(self, u):
        # u = [dx, dy, dyaw] expressed in the previous body frame
        dx, dy, dyaw = u
        theta = self.pose[2]

        # Nonlinear motion model
        self.pose[0] += dx * cos(theta) - dy * sin(theta)
        self.pose[1] += dx * sin(theta) + dy * cos(theta)
        self.pose[2] = wrap_angle(self.pose[2] + dyaw)

        # Jacobian of motion model w.r.t. state
        A = np.array([
            [1.0, 0.0, -dx * sin(theta) - dy * cos(theta)],
            [0.0, 1.0,  dx * cos(theta) - dy * sin(theta)],
            [0.0, 0.0, 1.0]
        ])

        self.S = A @ self.S @ A.T + self.R

    def predict_EKF(self, u):
        # Alias kept for compatibility with fusion.py
        self.predict(u)

    def update(self, z):
        # z = [gps_x, gps_y]
        H = self.C
        innovation = z - H @ self.pose

        K = self.S @ H.T @ np.linalg.inv(H @ self.S @ H.T + self.Q)
        self.pose = self.pose + K @ innovation
        self.pose[2] = wrap_angle(self.pose[2])

        # Joseph form keeps S symmetric positive semi-definite under FP noise
        I = np.identity(3)
        IKH = I - K @ H
        self.S = IKH @ self.S @ IKH.T + K @ self.Q @ K.T
        return self.pose, self.S
