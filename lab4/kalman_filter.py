import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.state = np.array([x, y, yaw], dtype=float)

        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)

        # State covariance matrix
        self.S = np.identity(3) * 1

        # Observation matrix: only x and y are measured (2x3)
        self.C = np.array([[1, 0, 0],
                           [0, 1, 0]], dtype=float)

        # control_variance
        c_var = 1
        # measurement_variance
        m_var = 3

        # State transition noise
        # control_variance = 1
        self.R = np.identity(3) * c_var

        # Measurement noise
        # measurement_variance = 3 
        self.Q = np.identity(2) * m_var
    def predict(self, u):
        # State prediction: x_k = A * x_{k-1} + B * u_k
        self.state = self.A @ self.state + self.B @ u
        # Covariance prediction: S = A * S * A^T + R
        self.S = self.A @ self.S @ self.A.T + self.R

    def update(self, z):
        # Kalman gain: K = S * C^T * (C * S * C^T + Q)^-1
        Kalman_gain = self.S @ self.C.T @ np.linalg.inv(self.C @ self.S @ self.C.T + self.Q)
        # State update: x = x + K * (z - C * x)
        self.state = self.state + Kalman_gain @ (z - self.C @ self.state)
        # Covariance update: S = (I - K * C) * S
        self.S = (np.identity(3) - Kalman_gain @ self.C) @ self.S
        return self.state, self.S
    