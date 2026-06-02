#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Path.h>


using namespace std;

ros::Publisher path_pub;
ros::Publisher radar_pose_pub;
nav_msgs::Path path;

ros::Publisher gt_path_pub;
ros::Publisher benchmark_pose_pub;
nav_msgs::Path gt_path;

void radar_pose_publisher(float x, float y, float z, float yaw)
{
    geometry_msgs::PoseStamped pose;
    tf2::Quaternion myQuaternion;
    myQuaternion.setRPY(0, 0, yaw);
    myQuaternion.normalize();

    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";

    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = z;

    pose.pose.orientation.x = myQuaternion.getX();
    pose.pose.orientation.y = myQuaternion.getY();
    pose.pose.orientation.z = myQuaternion.getZ();
    pose.pose.orientation.w = myQuaternion.getW();
    radar_pose_pub.publish(pose);
    
    path.header.frame_id = "map";
    pose.header.stamp = ros::Time::now();
    path.poses.push_back(pose);
    path_pub.publish(path);
}


void matrix4fToTfTransform(const Eigen::Matrix4f &matrix, tf::Transform &tf_transform) {
    tf::Matrix3x3 tf_rotation(
        matrix(0, 0), matrix(0, 1), matrix(0, 2),
        matrix(1, 0), matrix(1, 1), matrix(1, 2),
        matrix(2, 0), matrix(2, 1), matrix(2, 2)
    );
    tf::Vector3 tf_translation(matrix(0, 3), matrix(1, 3), matrix(2, 3));
    tf_transform.setBasis(tf_rotation);
    tf_transform.setOrigin(tf_translation);
    tf::Quaternion q = tf_transform.getRotation();
    q.normalize();
    tf_transform.setRotation(q);
}

int main(int argc, char** argv)
{
    ros::init (argc, argv, "map_visualization");
    ros::NodeHandle nh;
    ros::Publisher map_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/map", 1);
    ros::Publisher radar_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/radar", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/visual/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/visual/localization_path", 1);

    tf::TransformBroadcaster br;

    string map_file = "path_to_map.pcd";

    pcl::PointCloud<pcl::PointXYZI>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_file, *map_pc) == -1)
    {
        PCL_ERROR ("Couldn't read file\n");
    }

    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(map_pc);
    pass.setFilterFieldName("intensity");
    pass.setFilterLimits(10, 256);
    pass.setFilterLimitsNegative(false);
    pass.filter(*map_pc);

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*map_pc, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    
    map_pub.publish(map_msg);

    ros::Duration(3).sleep();

    string t_file = "path_to_radar_t_list.txt";  
    //radar_pose
    string pose_file = "path_to_result.txt";
    string line;

    vector<string> t_list;

    ifstream t_in(t_file);
    ifstream pose_in(pose_file);


    if(!t_in.is_open())
    {
        cout << "Error opening t file" << endl;
        return 0;
    }

    while(getline(t_in, line) && ros::ok())
    {
        string pcd_file = "path_to_pcd_folder/" + line + ".pcd";cout << line << endl;
        t_list.push_back(pcd_file);
    }  

    if(!pose_in.is_open())
    {
        cout << "Error opening pose file" << endl;
        return 0;
    }

    int idx = 0;

    while(getline(pose_in, line) && ros::ok())
    {
        string s;
        stringstream ss(line);
        ss.str(line);
        vector<string> record;
        while(getline(ss, s, ' '))
        {
            record.push_back(s);
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(t_list[idx], *radar_pc) == -1)
        {
            PCL_ERROR ("Couldn't read file\n");
        }
        ROS_WARN("%s", record[0].c_str());

        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

        transform << stof(record[0]), stof(record[1]), stof(record[2]), stof(record[3]), 
                     stof(record[4]), stof(record[5]), stof(record[6]), stof(record[7]), 
                     stof(record[8]), stof(record[9]), stof(record[10]), stof(record[11]),
                     0, 0, 0, 1;




        sensor_msgs::PointCloud2 radar_msg;

        Eigen::Matrix4f transform2 = transform;
        tf::Transform tf_transform2;
        matrix4fToTfTransform(transform2, tf_transform2);
        br.sendTransform(tf::StampedTransform(tf_transform2, ros::Time::now(), "map", "radar"));                     
        pcl::toROSMsg(*radar_pc, radar_msg);
        radar_msg.header.frame_id = "radar";
        radar_msg.header.stamp = ros::Time::now();
        radar_pub.publish(radar_msg);

        //publish the path
        radar_pose_publisher(transform(0, 3), transform(1, 3), transform(2, 3), atan2(transform(1, 0), transform(0, 0)));


        idx++;
        ros::Duration(0.02).sleep();

    }
    ros::spin(); 
    return 0;
}


