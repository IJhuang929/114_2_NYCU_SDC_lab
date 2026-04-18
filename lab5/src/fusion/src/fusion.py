#!/usr/bin/env python3

import rospy
import os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from nav_msgs.msg import Odometry
from math import cos, sin, atan2, sqrt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from EKF import ExtendedKalmanFilter

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.EKF = None
        
        self.gt_list = [[], []]
        self.est_list = [[], []]
        self.initial = False
        # You may need to store the last odometry data for calculating control input
        self.last_odom = None

    def shutdown(self):
        print("shuting down fusion.py")

    def predictPublish(self):
        
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        
        # new
        predPose.header.stamp = rospy.Time.now()
        x, y, yaw = self.EKF.pose
        
        # change to the state x and state y from EKF
        predPose.pose.pose.position.x = x
        predPose.pose.pose.position.y = y
        
        # Change to the state yaw from EKF
        quaternion = quaternion_from_euler(0, 0, yaw)
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        
        # Change to the covariance matrix of [x, y, yaw] from EKF
        predPose.pose.covariance = [self.EKF.S[0,0], self.EKF.S[0,1], 0, 0, 0, self.EKF.S[0,2],
                                    self.EKF.S[1,0], self.EKF.S[1,1], 0, 0, 0, self.EKF.S[1,2],
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    self.EKF.S[2,0], self.EKF.S[2,1], 0, 0, 0, self.EKF.S[2,2]]
                                    
        self.posePub.publish(predPose)
    
    def odometryCallback(self, data):
        odom_x = data.pose.pose.position.x
        odom_y = data.pose.pose.position.y
        odom_quaternion = [
            data.pose.pose.orientation.x, data.pose.pose.orientation.y,
            data.pose.pose.orientation.z, data.pose.pose.orientation.w
        ]
        _, _, odom_yaw = euler_from_quaternion(odom_quaternion)
        odom_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Design the control of EKF state from radar odometry data
        # The data is in global frame, you may need to find a way to convert it into local frame
        # Ex. 
        #     Find differnence between 2 odometry data 
        #         -> diff_x = ???
        #         -> diff_y = ???
        #         -> diff_yaw = ???
        #     Calculate transformation matrix between 2 odometry data
        #         -> transformation = last_odom_pose^-1 * current_odom_pose
        #     etc.
        if not self.initial:
            self.EKF = ExtendedKalmanFilter(odom_x, odom_y, odom_yaw)
            self.last_odom = (odom_x, odom_y, odom_yaw)
            self.initial = True

        else:
            # Local-frame increment between last and current odom (global-frame poses)
            prev_x, prev_y, prev_yaw = self.last_odom
            dx_global = odom_x - prev_x
            dy_global = odom_y - prev_y
            dx_local =  dx_global * cos(prev_yaw) + dy_global * sin(prev_yaw)
            dy_local = -dx_global * sin(prev_yaw) + dy_global * cos(prev_yaw)
            d_yaw = atan2(sin(odom_yaw - prev_yaw), cos(odom_yaw - prev_yaw))

            control = [dx_local, dy_local, d_yaw]

            # Step-wise process noise: scale with motion so GPS retains influence
            dist = sqrt(dx_local * dx_local + dy_local * dy_local)
            sigma_xy = 0.05 + 0.05 * dist
            sigma_yaw = 0.5 * 3.14159265 / 180.0
            self.EKF.R = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
            self.EKF.predict(u = control)
            
        self.last_odom = (odom_x, odom_y, odom_yaw)
        self.predictPublish()
        
    def gpsCallback(self, data):
        if not self.initial: return
        
        gps_x = data.pose.pose.position.x
        gps_y = data.pose.pose.position.y
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Design the measurement of EKF state from GPS data
        # Ex. 
        #     Use GPS directly
        #     Find a approximate yaw
        #     etc.
        measurement = np.array([gps_x, gps_y])

        # Update error covriance
        self.EKF.Q = np.array(data.pose.covariance).reshape(6,6)[:2,:2]
        self.EKF.update(z = measurement)
        self.predictPublish()
    
    def gtCallback(self, data):
        self.gt_list[0].append(data.pose.pose.position.x)
        self.gt_list[1].append(data.pose.pose.position.y)
        if self.EKF is not None:
            # Change to the state x and state y from EKF
            self.est_list[0].append(self.EKF.pose[0])
            self.est_list[1].append(self.EKF.pose[1])
    
        return

    def plot_path(self):
        gt = np.array(self.gt_list)
        est = np.array(self.est_list)
        n = min(gt.shape[1], est.shape[1])
        if n > 0:
            err = np.linalg.norm(gt[:, :n] - est[:, :n], axis=0)
            rmse = float(np.sqrt((err ** 2).mean()))
            print(f"[EKF] samples={n}  RMSE={rmse:.3f} m  max={err.max():.3f} m")

        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.plot(self.gt_list[0], self.gt_list[1], alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(self.est_list[0], self.est_list[1], alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("/root/catkin_ws/result"):
            print("not exist")
            os.mkdir("/root/catkin_ws/result")
        plt.savefig("/root/catkin_ws/result/result2.png")
        return
    
if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    fusion.plot_path()
