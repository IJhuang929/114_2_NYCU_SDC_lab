import rospy
import math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray


MAX_DISTANCE = 20.0     # don't visualize objects beyond 20m

class RadarRelativeSpeedNode:
    def __init__(self):
        rospy.init_node('radar_relative_speed_node', anonymous=True)
        
        # 1. subscribe radar_front objects topic 
        self.sub = rospy.Subscriber('/ars548/radar_front/objects', PointCloud2, self.callback)
        
        # 2. publisher for visualization markers
        self.marker_pub = rospy.Publisher('/radar_speed_markers', MarkerArray, queue_size=10)
        
        rospy.loginfo("Radar Relative Speed Node 啟動成功！")

    def callback(self, data):
        # check if there are subscribers to the marker topic
        fields_in_msg = [f.name for f in data.fields]
        # msg name
        target_v_x = "DynamicsRelVelX"
        target_v_y = "DynamicsRelVelY"
        target_v_abs_x = "DynamicsAbsVelX"
        target_v_abs_y = "DynamicsAbsVelY"

      
        if target_v_x not in fields_in_msg or target_v_y not in fields_in_msg or target_v_abs_x not in fields_in_msg or target_v_abs_y not in fields_in_msg not in fields_in_msg:
            rospy.logwarn_once(f"Available fields: {fields_in_msg}")
            return
        
        # initialize MarkerArray
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        
        read_fields = ("x", "y", "z", target_v_abs_x, target_v_abs_y, target_v_x, target_v_y)
        
        try:
            
            points = list(pc2.read_points(data, field_names=read_fields, skip_nans=True))
            
            for i, p in enumerate(points):
                pt_dict = dict(zip(read_fields, p))
                x = pt_dict['x']
                y = pt_dict['y']
                z = pt_dict['z']
                rel_v_x = pt_dict[target_v_x]
                rel_v_y = pt_dict[target_v_y]
                abs_v_x = pt_dict[target_v_abs_x]
                abs_v_y = pt_dict[target_v_abs_y]

                distance = math.sqrt(x**2 + y**2)
                abs_speed = math.sqrt(abs_v_x**2 + abs_v_y**2)
                rel_speed = x * rel_v_x + y * rel_v_y / distance
                # lower than 1.5 m/s or beyond MAX_DISTANCE, skip visualization
                if abs_speed < 1.5: continue
                if distance > MAX_DISTANCE: continue

                # create visualization marker
                marker = Marker()
                marker.header = data.header
                marker.ns = "rel_speed"
                marker.id = i
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                
                # set marker position (slightly above the object)
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z + 1.0 
                marker.pose.orientation.w = 1.0
                
                # set text format
                marker.text = f"ID:{i} V:{rel_speed:.1f}m/s"
                
                # set appearance
                marker.scale.z = 0.7
                marker.color.a = 1.0
                
                # color code: red for approaching, green for receding
                if rel_speed < 0:    
                    marker.color.r, marker.color.g, marker.color.b = (1.0, 0.2, 0.2)
                else:
                    marker.color.r, marker.color.g, marker.color.b = (0.2, 1.0, 0.2)
                
                marker.lifetime = rospy.Duration(0.1)
                marker_array.markers.append(marker)

           
            self.marker_pub.publish(marker_array)
                
        except ValueError as e:
            rospy.logerr(f"Data parsing error: {e}")

if __name__ == '__main__':
    try:
        node = RadarRelativeSpeedNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
