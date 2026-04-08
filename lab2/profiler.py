import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np

def callback(data):
    # get Topic Name
    topic_name = data._connection_header['topic']
    
    # get Timestamp
    timestamp = data.header.stamp.to_sec()
    
    # get X, Y, Z
    points = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)))
    
    if points.size == 0:
        rospy.loginfo(f"Topic: {topic_name} - No points received.")
        return

    # get number of points and ranges
    num_points = len(points)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    # print the results
    print("-" * 30)
    print(f"Topic Name: {topic_name}")
    print(f"Timestamp: {timestamp:.3f}")
    print(f"Total number of points: {num_points}")
    print(f"X range: [{x_min:.1f}, {x_max:.1f}] m")
    print(f"Y range: [{y_min:.1f}, {y_max:.1f}] m")
    print(f"Z range: [{z_min:.1f}, {z_max:.1f}] m")

def listener():
    rospy.init_node('profiler', anonymous=True)

    rospy.Subscriber("/ouster/top_122219002200", PointCloud2, callback)
    rospy.Subscriber("/ars548/radar_front/detections", PointCloud2, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()