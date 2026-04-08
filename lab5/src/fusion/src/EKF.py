import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self, x, y, yaw):
        # Define what state to be estimate
        # Ex.
        #   only pose -> np.array([x, y, yaw])
        #   with velocity -> np.array([x, y, yaw, vx, vy, vyaw])
        #   etc...
        # self.pose = ???
        self.pose = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.identity(3)
        
        # State transition error
        self.R = np.identity(3) * 1
        
        # Measurement error
        self.Q = np.identity(3) * 1
        print("Initialize Kalman Filter")
    
    def predict(self, u):
        # Base on the Kalman Filter design in Assignment 4
        # Implement a linear or nonlinear motion model for the control input
        # Calculate Jacobian matrix of the model as self.A
        
        dx, dy, dyaw = u
        theta = self.pose[2]

        # 1. 更新狀態預測 (Nonlinear Motion Model)
        self.pose[0] += dx * cos(theta) - dy * sin(theta)
        self.pose[1] += dx * sin(theta) + dy * cos(theta)
        self.pose[2] += dyaw

        # 2. 計算 Jacobian A
        A = np.array([
            [1, 0, -dx * sin(theta) - dy * cos(theta)],
            [0, 1,  dx * cos(theta) - dy * sin(theta)],
            [0, 0, 1]
        ])

        # 3. 更新協方差預測
        self.S = A @ self.S @ A.T + self.R
    
        
    def update(self, z):
        # Base on the Kalman Filter design in Assignment 4
        # Implement a linear or nonlinear observation matrix for the measurement input
        # Calculate Jacobian matrix of the matrix as self.C
        
        # z = [gps_x, gps_y]
        # 1. 計算 Kalman Gain
        H = self.C
        K = self.S @ H.T @ np.linalg.inv(H @ self.S @ H.T + self.Q)

        # 2. 更新狀態
        innovation = z - H @ self.pose
        self.pose = self.pose + K @ innovation

        # 3. 更新協方差
        self.S = (np.identity(3) - K @ H) @ self.S
        return self.pose, self.S
    
    
    
        