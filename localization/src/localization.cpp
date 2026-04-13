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
#include <deque>
#include <pcl/filters/radius_outlier_removal.h>

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

void callback();
void matrix4fToTfTransform(const Eigen::Matrix4f &matrix, tf::Transform &tf_transform);
void radar_pose_publisher(float x, float y, float z, float yaw);

pcl::PointCloud<pcl::PointXYZI>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);

Eigen::Matrix4f transform_matrix;
int frame_count = 0;

// === 恆定速度模型 (CV Model) 與 歷史狀態變數 ===
Eigen::Matrix4f prev_pose = Eigen::Matrix4f::Identity();
Eigen::Matrix4f prev_prev_pose = Eigen::Matrix4f::Identity();
bool has_prev_prev = false;
bool initialized = false;

// === 雷達 Buffer ===
struct FrameData {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Eigen::Matrix4f pose;
};
std::deque<FrameData> radar_buffer;
const int BUFFER_SIZE = 5; // 決定要疊加過去幾幀雷達點

// modify IO path
string t_file = "/root/catkin_ws/src/localization/data/Track1/radar_t_list.txt";
// string pcd_folder = "/root/catkin_ws/src/localization/data/Track1/stack_3_frame_radar_pcd";
string pcd_folder = "/root/catkin_ws/src/localization/data/Track1/radar_pcd_with_intensity";
string map_file = "/root/catkin_ws/src/localization/data/Track1/test.pcd";
string save_dir = "/root/catkin_ws/src/localization/result/";

int main(int argc, char** argv) {

    ros::init(argc, argv, "radar_pcd_publisher");
    ros::NodeHandle nh;

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar", 1);
    submap_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar_submap_pc", 1);
    map_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pc", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/localization_path", 1);

    time_t now = time(0);
    tm* t = localtime(&now);
    ostringstream filename;
    filename << save_dir << "result_"
             << (t->tm_year + 1900)
             << setfill('0') << setw(2) << (t->tm_mon + 1)
             << setw(2) << t->tm_mday << "_"
             << setw(2) << t->tm_hour
             << setw(2) << t->tm_min
             << setw(2) << t->tm_sec << ".txt";
    string save_file = filename.str();
    ROS_INFO("Saving results to: %s", save_file.c_str());
    file.open(save_file);

    tf::TransformBroadcaster br;

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (map_file, *map_pc) == -1)
    {
        PCL_ERROR ("Couldn't read file pcd \n");
    }

    ifstream t_in(t_file);
    string line;

    while(getline(t_in, line) && ros::ok())
    {
        string pcd_file = pcd_folder + "/" + line + ".pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *radar_pc) == -1)
        {
            PCL_ERROR ("Couldn't read file\n");
        }

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
    ros::spin();
    return 0;
}

void callback(){
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

    // ====================================================================
    // === 恆定速度模型預測 (CV Model) ===
    // ====================================================================
    if(!initialized){
        ifstream intial_pose_file("/root/catkin_ws/src/localization/data/Track1/init_guess.txt");
        if (intial_pose_file.is_open()) {
            for(int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    char comma;
                    intial_pose_file >> transform_matrix(i, j);
                    if (j < 3) intial_pose_file >> comma;
                }
            }
        } else {
             ROS_WARN("Cannot open init_guess.txt, using Identity.");
             transform_matrix = Eigen::Matrix4f::Identity();
        }
        init_guess = transform_matrix;
        prev_pose = transform_matrix;
        initialized = true;
    }
    else{
        if (has_prev_prev) {
            Eigen::Matrix4f delta_pose = prev_prev_pose.inverse() * prev_pose;
            init_guess = prev_pose * delta_pose; 
        } else {
            init_guess = prev_pose;
        }
    }

    // ====================================================================
    // === 地圖裁切 (Submap) ===
    // ====================================================================
    float center_x = init_guess(0, 3);
    float center_y = init_guess(1, 3);
    float square_size = 200.0;      
    float half_size = square_size / 2.0;
    Eigen::Vector4f min_pt(center_x - half_size, center_y - half_size, -std::numeric_limits<float>::max(), 1.0);
    Eigen::Vector4f max_pt(center_x + half_size, center_y + half_size, std::numeric_limits<float>::max(), 1.0);
                
    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
    crop_box_filter.setInputCloud(map_pc);
    crop_box_filter.setMin(min_pt);
    crop_box_filter.setMax(max_pt);
    crop_box_filter.filter(*submap_pc);

    // ====================================================================
    // === 雷達點雲基本過濾 ===
    // ====================================================================
    pcl::PointCloud<pcl::PointXYZI>::Ptr radar_pc_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    // 設定強度過濾門檻 (此數值建議從 5.0 ~ 10.0 開始測試)
    const float INTENSITY_THRESHOLD = 8.0;

    for (const auto& pt : radar_pc->points) {
        float dist = sqrt(pt.x * pt.x + pt.y * pt.y);
        if (dist > 5.0 && dist < 60.0) {
            pcl::PointXYZI filter_pt;
            filter_pt.x = pt.x;
            filter_pt.y = pt.y;
            filter_pt.z = pt.z;
            //實作論文巧思：雷達強度距離補償 Pn = log(R^4 * Pr), 避免 R^4 數值過大溢位，我們改用 4 * log(R) + log(Pr)
            float Pr = std::max(pt.intensity, 1e-6f);
            float compensated_intensity = 4.0f * log(dist) + log(Pr);

            // 3. 濾除補償後強度過低的點 (過濾掉雜訊與非建築物)
            if (compensated_intensity > INTENSITY_THRESHOLD) {
                filter_pt.intensity = compensated_intensity;
                radar_pc_filtered->points.push_back(filter_pt);
            }
            
            
        }
    }

    pcl::VoxelGrid<pcl::PointXYZI> radar_voxel;
    radar_voxel.setInputCloud(radar_pc_filtered);
    radar_voxel.setLeafSize(0.5, 0.5, 0.5);
    radar_voxel.filter(*radar_pc_filtered);

    // ====================================================================
    // === 多幀雷達點雲疊加 (Frame Stacking) ===
    // ====================================================================
    pcl::PointCloud<pcl::PointXYZI>::Ptr stacked_radar_pc(new pcl::PointCloud<pcl::PointXYZI>);
    *stacked_radar_pc += *radar_pc_filtered;

    for (const auto& frame : radar_buffer) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_history(new pcl::PointCloud<pcl::PointXYZI>);
        Eigen::Matrix4f T_history_to_current = init_guess.inverse() * frame.pose;
        pcl::transformPointCloud(*(frame.cloud), *transformed_history, T_history_to_current);
        *stacked_radar_pc += *transformed_history;
    }

    // ====================================================================
    // === 雷達雜訊過濾 (Radius Outlier Removal) ===
    // ====================================================================
    pcl::PointCloud<pcl::PointXYZI>::Ptr cleaned_radar_pc(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror;
    ror.setInputCloud(stacked_radar_pc);
    ror.setRadiusSearch(3.0);       
    ror.setMinNeighborsInRadius(2); 
    ror.filter(*cleaned_radar_pc);

    // ====================================================================
    // === Coarse-to-Fine NDT 配準 (3D) ===
    // ====================================================================
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setTransformationEpsilon(1e-3);
    ndt.setStepSize(0.1);
    ndt.setInputTarget(submap_pc);
    ndt.setInputSource(cleaned_radar_pc);

    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_scan(new pcl::PointCloud<pcl::PointXYZI>);
    
    // 第一階段：大網格 (Coarse)
    ndt.setResolution(3.0); 
    ndt.setMaximumIterations(35);
    ndt.align(*aligned_scan, init_guess);
    Eigen::Matrix4f ndt_result = ndt.getFinalTransformation();

    // 第二階段：小網格 (Fine)
    if (ndt.hasConverged()) {
        ndt.setResolution(1.0);
        ndt.setMaximumIterations(30);
        ndt.align(*aligned_scan, ndt_result); 
        ndt_result = ndt.getFinalTransformation();
    }

    // ====================================================================
    // === 發散保護 ===
    // ====================================================================
    float dx = ndt_result(0, 3) - init_guess(0, 3);
    float dy = ndt_result(1, 3) - init_guess(1, 3);
    float jump_distance = sqrt(dx * dx + dy * dy);

    if (jump_distance > 3.0 || !ndt.hasConverged()) {
        ROS_WARN("NDT Diverged or Jumped Too Far! Using CV Model Prediction.");
        transform_matrix = init_guess; 
    } else {
        transform_matrix = ndt_result;
    }
    transform_matrix(2, 3) = init_guess(2, 3);

    // ====================================================================
    // === 更新狀態變數與 Buffer ===
    // ====================================================================
    prev_prev_pose = prev_pose;
    has_prev_prev = true;
    prev_pose = transform_matrix;

    FrameData current_frame;
    current_frame.cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(*radar_pc_filtered));
    current_frame.pose = transform_matrix;
    radar_buffer.push_front(current_frame);
    if (radar_buffer.size() > BUFFER_SIZE) {
        radar_buffer.pop_back(); 
    }

    // ====================================================================
    // === 發布與記錄 ===
    // ====================================================================
    frame_count++;
    float yaw = atan2(transform_matrix(1, 0), transform_matrix(0, 0));
    ROS_INFO("Frame %d | NDT Score: %.4f | x: %.2f y: %.2f | Buffer: %lu", 
             frame_count, ndt.getFitnessScore(), transform_matrix(0,3), transform_matrix(1,3), radar_buffer.size());

    radar_pose_publisher(transform_matrix(0, 3), transform_matrix(1, 3),  transform_matrix(2, 3), yaw);
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

void radar_pose_publisher(float x, float y, float z, float yaw) {
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