#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <tf/transform_listener.h>

using namespace std;

ros::Publisher pc_pub;
ros::Publisher submap_pc_pub;
ros::Publisher map_pc_pub;
ros::Publisher radar_pose_pub;
ros::Publisher path_pub;
nav_msgs::Path path;
ofstream file;
ofstream diag_file;
int frame_count = 0;

void callback();
void matrix4fToTfTransform(const Eigen::Matrix4f &matrix, tf::Transform &tf_transform);
void radar_pose_publisher(float x, float y, float z, float yaw);

pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

bool initialized = false;
Eigen::Matrix4f transform_matrix;

// modify IO path
string t_file = "/root/catkin_ws/src/localization/data/Track1/radar_t_list.txt";
string pcd_folder = "/root/catkin_ws/src/localization/data/Track1/stack_3_frame_radar_pcd";
string map_file = "/root/catkin_ws/src/localization/data/Track1/test.pcd";
string save_dir = "/root/catkin_ws/src/localization/result/";

// === radar_front → base_link 靜態外參 ===
Eigen::Matrix4f T_radar_to_base;

int main(int argc, char** argv) {

    ros::init(argc, argv, "radar_pcd_publisher");
    ros::NodeHandle nh;

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar", 1);
    submap_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar_submap_pc", 1);
    map_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pc", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/localization_path", 1);

    // === result帶時間戳 ===
    time_t now = time(0);
    tm* t = localtime(&now);
    ostringstream result_fn;
    result_fn << save_dir << "result_"
              << (t->tm_year + 1900)
              << setfill('0') << setw(2) << (t->tm_mon + 1)
              << setw(2) << t->tm_mday << "_"
              << setw(2) << t->tm_hour
              << setw(2) << t->tm_min
              << setw(2) << t->tm_sec << ".txt";
    file.open(result_fn.str());
    ROS_INFO("Saving results to: %s", result_fn.str().c_str());

    ostringstream diag_fn;
    diag_fn << save_dir << "diag_"
            << (t->tm_year + 1900)
            << setfill('0') << setw(2) << (t->tm_mon + 1)
            << setw(2) << t->tm_mday << "_"
            << setw(2) << t->tm_hour
            << setw(2) << t->tm_min
            << setw(2) << t->tm_sec << ".txt";
    diag_file.open(diag_fn.str());
    diag_file << "frame,score,x,y,yaw,radar_pts,submap_pts,converged" << endl;
    ROS_INFO("Saving diagnostics to: %s", diag_fn.str().c_str());

    tf::TransformBroadcaster br;

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (map_file, *map_pc) == -1)
    {
        PCL_ERROR ("Couldn't read file pcd \n");
        return -1;
    }
    ROS_INFO("Map loaded: %lu points", map_pc->points.size());

    // === 計算外參 ===
    Eigen::Matrix4f T_base_to_radar = Eigen::Matrix4f::Identity();
    T_base_to_radar <<  0.999983,  0.00505, -0.002915,  2.022,
                       -0.005,     0.999843,  0.017014,  0.359,
                        0.003,    -0.016999,  0.999851, -1.15,
                        0.0,       0.0,       0.0,       1.0;
    T_radar_to_base = T_base_to_radar.inverse();

    ifstream t_in(t_file);
    string line;

    while(getline(t_in, line) && ros::ok())
    {
        string pcd_file = pcd_folder + "/" + line + ".pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *radar_pc) == -1)
        {
            PCL_ERROR ("Couldn't read file\n");
            continue;
        }

        // 將 radar 點雲從 radar_front frame 轉換到 base_link frame
        pcl::transformPointCloud(*radar_pc, *radar_pc, T_radar_to_base);

        callback();

        sensor_msgs::PointCloud2 submap_msg;
        pcl::toROSMsg(*submap_pc, submap_msg);
        submap_msg.header.frame_id = "map";
        submap_msg.header.stamp = ros::Time::now();
        submap_pc_pub.publish(submap_msg);

        tf::Transform tf_transform1;
        matrix4fToTfTransform(transform_matrix, tf_transform1);
        br.sendTransform(tf::StampedTransform(tf_transform1, ros::Time::now(), "map", "radar"));

        sensor_msgs::PointCloud2 radar_msg;
        pcl::toROSMsg(*radar_pc, radar_msg);
        radar_msg.header.frame_id = "radar";
        radar_msg.header.stamp = ros::Time::now();
        pc_pub.publish(radar_msg);
    }
    diag_file << "# total_frames," << frame_count << endl;
    diag_file.close();
    file.close();
    ROS_INFO("Done. Total frames: %d", frame_count);
    ros::spin();
    return 0;
}

void callback(){
    Eigen::Matrix4f init_guess;

    // Update pose
    if(!initialized){
        init_guess << -0.7739014871732518, -0.632632984298013, 0.029188959029395573, -25.31523704528809,
                      0.632582897383501, -0.7743982245509299, -0.012094120483231403, -4.389283657073975,
                      0.030255017582620166, 0.009104778446398086, 0.9995007438318981, -7.73,
                      0.0,      0.0,      0.0,    1.0;
        initialized = true;
    }
    else{
        init_guess = transform_matrix;
    }

    // dynamic crop map into submap to accelerate scan matching
    float center_x = init_guess(0, 3);
    float center_y = init_guess(1, 3);
    float square_size = 400.0;

    float half_size = square_size / 2.0;
    Eigen::Vector4f min_pt(center_x - half_size, center_y - half_size, -std::numeric_limits<float>::max(), 1.0);
    Eigen::Vector4f max_pt(center_x + half_size, center_y + half_size, std::numeric_limits<float>::max(), 1.0);

    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
    crop_box_filter.setInputCloud(map_pc);
    crop_box_filter.setMin(min_pt);
    crop_box_filter.setMax(max_pt);
    crop_box_filter.filter(*submap_pc);

    ndt.setInputSource(radar_pc);
    ndt.setInputTarget(submap_pc);
    ndt.setTransformationEpsilon(0.001);
    ndt.setStepSize(0.1);
    ndt.setResolution(2.0);
    ndt.setMaximumIterations(50);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_pc(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*aligned_pc, init_guess);

    if (ndt.hasConverged()) {
        transform_matrix = ndt.getFinalTransformation();
    } else {
        ROS_WARN("NDT did not converge.");
        transform_matrix = init_guess;
    }

    // === Log & Diag ===
    frame_count++;
    float score = ndt.getFitnessScore();
    float x = transform_matrix(0, 3);
    float y = transform_matrix(1, 3);
    float z = transform_matrix(2, 3);
    float yaw = atan2(transform_matrix(1, 0), transform_matrix(0, 0));
    int converged = ndt.hasConverged() ? 1 : 0;

    ROS_INFO("Frame %d | Score: %.2f | x: %.2f y: %.2f z: %.2f | yaw: %.3f | radar: %lu submap: %lu",
             frame_count, score, x, y, z, yaw, radar_pc->points.size(), submap_pc->points.size());

    diag_file << frame_count << "," << score << "," << x << "," << y << "," << yaw
              << "," << radar_pc->points.size() << "," << submap_pc->points.size()
              << "," << converged << endl;

    radar_pose_publisher(x, y, z, yaw);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            if(i == 2 && j == 3){
                file << transform_matrix(i, j) << endl;
            }
            else
                file << transform_matrix(i, j) << " ";
        }
    }

    file.flush();
    return;
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
