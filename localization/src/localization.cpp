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


#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Path.h>

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

pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;


bool initialized = false;
Eigen::Matrix4f transform_matrix;
int frame_count = 0;

int reject_count = 0;       // 連續 reject 計數

// --- [新增] 恆速模型 (CVM) 所需的歷史位姿 ---
Eigen::Matrix4f prev_pose = Eigen::Matrix4f::Identity();
Eigen::Matrix4f prev_prev_pose = Eigen::Matrix4f::Identity();
int pose_history_count = 0;

// 直線偵測相關
const int STRAIGHT_HISTORY_SIZE = 10;       // 觀察最近幾幀
std::vector<float> yaw_history;              // 歷史 yaw 值
bool is_straight_line = false;
const float STRAIGHT_YAW_THRESHOLD = 0.5;  // 度，N 幀內 yaw 變化量的標準差低於此值判定為直線

// modify IO path
string t_file = "/root/catkin_ws/src/localization/data/Track1/radar_t_list.txt";
string pcd_folder = "/root/catkin_ws/src/localization/data/Track1/stack_3_frame_radar_pcd";
// string map_file = "/root/catkin_ws/src/localization/data/Track1/map.pcd";
string map_file = "/root/catkin_ws/src/localization/data/Track1/test.pcd";
string save_dir = "/root/catkin_ws/src/localization/result/";


Eigen::Affine3d radar_ex = Eigen::Affine3d::Identity();

int main(int argc, char** argv) {

    ros::init(argc, argv, "radar_pcd_publisher");
    ros::NodeHandle nh;

    // Publisher for the point cloud
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar", 1);
    submap_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar_submap_pc", 1);
    map_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pc", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/localization_path", 1);

    // Generate timestamped filename: result_YYYYMMDD_HHMMSS.txt
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

    // Set NDT parameters once
    ndt.setTransformationEpsilon(1e-3);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setMaximumIterations(80);

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
    Eigen::Matrix4f init_guess;
//-------------------------------------------------------------------------------------------------
    // Update pose
    if(!initialized){
        // read initial pose from file, if not exist, use identity matrix as initial guess
        ifstream intial_pose_file("/root/catkin_ws/src/localization/data/Track1/init_guess.txt");
        for(int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                char comma;
                intial_pose_file >> transform_matrix(i, j);
                if (j < 3) {
                    intial_pose_file >> comma; // read the comma
                }
            }
        }
        init_guess = transform_matrix;
        //  初始化位姿歷史
        prev_pose = transform_matrix;
        pose_history_count = 1;
        initialized = true;
    }
    else{
        if (pose_history_count >= 2) {
            // 計算上上一幀到上一幀的變化矩陣 (速度)
            Eigen::Matrix4f delta_transform = prev_prev_pose.inverse() * prev_pose;
            // 將這個變化量套用到上一幀，預測當前幀的 Initial Guess
            init_guess = prev_pose * delta_transform;
        } else {
            init_guess = prev_pose;
        }
    }
//-------------------------------------------------------------------------------------------------
    
    // dynamic crop map into submap to accelerate scan matching
    float center_x = init_guess(0, 3);
    float center_y = init_guess(1, 3);
    float square_size = 100.0;      // 400 to 100

    float half_size = square_size / 2.0;
    Eigen::Vector4f min_pt(center_x - half_size, center_y - half_size, -std::numeric_limits<float>::max(), 1.0);
    Eigen::Vector4f max_pt(center_x + half_size, center_y + half_size, std::numeric_limits<float>::max(), 1.0);
                
    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
    crop_box_filter.setInputCloud(map_pc);
    crop_box_filter.setMin(min_pt);
    crop_box_filter.setMax(max_pt);
    crop_box_filter.filter(*submap_pc);

    // Downsample submap to reduce computation
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(submap_pc);
    voxel_filter.setLeafSize(1.0, 1.0, 1.0);        // 0.5 to 1.0, adjust based on map density and desired accuracy
    voxel_filter.filter(*submap_pc);

    // Filter radar points by distance: keep 5m ~ 60m range (remove nearby moving objects and far noisy points)
    pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& pt : radar_pc->points) {
        float dist = sqrt(pt.x * pt.x + pt.y * pt.y);
        if (dist > 5.0 && dist < 60.0) {
            radar_pc_filtered->points.push_back(pt);
        }
    }

    // Flatten z to 0 for NDT point clouds (force 2D matching)
    // Use copies to preserve original point clouds for RViz display
    pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc_flat(new pcl::PointCloud<pcl::PointXYZ>(*radar_pc_filtered));
    pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc_flat(new pcl::PointCloud<pcl::PointXYZ>(*submap_pc));
    for (auto& pt : submap_pc_flat->points) pt.z = 0.0;
    for (auto& pt : radar_pc_flat->points) pt.z = 0.0;

    // Also flatten z in init_guess for consistency
    Eigen::Matrix4f init_guess_flat = init_guess;
    init_guess_flat(2, 3) = 0.0;

//-------------------------------------------------------
    // scan matching
    ndt.setInputSource(radar_pc_flat);       // current scan as source
    ndt.setInputTarget(submap_pc_flat);      // current submap as target

    // align the current scan to the submap
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_scan(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*aligned_scan, init_guess_flat);


    // update initial guess for next iteration
    Eigen::Matrix4f ndt_result = ndt.getFinalTransformation();

    //reject outliers if jump too far or score too high
    float dx = ndt_result(0, 3) - init_guess(0, 3);
    float dy = ndt_result(1, 3) - init_guess(1, 3);
    float jump_distance = sqrt(dx * dx + dy * dy);
    float ndt_score = ndt.getFitnessScore();
    float dyaw = atan2(ndt_result(1, 0), ndt_result(0, 0)) - atan2(init_guess(1, 0), init_guess(0, 0));
    // normalize dyaw to [-pi, pi]
    while (dyaw > M_PI) dyaw -= 2 * M_PI;
    while (dyaw < -M_PI) dyaw += 2 * M_PI;

    // Check if moving backward: dot product of movement vector and heading direction
    float heading_yaw = atan2(init_guess(1, 0), init_guess(0, 0));
    float heading_x = cos(heading_yaw);
    float heading_y = sin(heading_yaw);
    float dot = dx * heading_x + dy * heading_y;
    bool moving_backward = (dot < -0.1) && (jump_distance > 0.3);  // 有明顯位移且方向相反

    bool accepted = true;
    if (jump_distance > 3.0 || !ndt.hasConverged() || fabs(dyaw) > 0.2 || ndt_score > 5.0 || moving_backward) {
        reject_count++;
        // 連續 reject 超過 3 次，強制接受 NDT 結果，避免永遠卡死
        if (reject_count >= 3 && ndt.hasConverged() && !moving_backward) {
            // 強制接受，但限制最大跳躍幅度
            float clamp_dist = 1.0;  // 最多跳 1m
            if (jump_distance > clamp_dist) {
                float scale = clamp_dist / jump_distance;
                ndt_result(0, 3) = init_guess(0, 3) + dx * scale;
                ndt_result(1, 3) = init_guess(1, 3) + dy * scale;
            }
            // 限制 yaw 變化
            float clamp_dyaw = 0.1;  // 最多轉 ~5.7 deg
            if (fabs(dyaw) > clamp_dyaw) {
                float prev_yaw_angle = atan2(init_guess(1, 0), init_guess(0, 0));
                float clamped_yaw = prev_yaw_angle + (dyaw > 0 ? clamp_dyaw : -clamp_dyaw);
                ndt_result(0, 0) = cos(clamped_yaw); ndt_result(0, 1) = -sin(clamped_yaw);
                ndt_result(1, 0) = sin(clamped_yaw); ndt_result(1, 1) = cos(clamped_yaw);
            }
            ROS_WARN("Force accept (clamped) after %d rejects: jump=%.2f->%.2f, score=%.2f, dyaw=%.2f deg",
                     reject_count, jump_distance, min(jump_distance, clamp_dist), ndt_score, dyaw * 180.0 / M_PI);
            transform_matrix = ndt_result;
            transform_matrix(2, 3) = init_guess(2, 3);
            reject_count = 0;
        } else {
            ROS_WARN("Frame rejected (%d): jump=%.2f, score=%.2f, dyaw=%.2f deg, converged=%d, backward=%d",
                     reject_count, jump_distance, ndt_score, dyaw * 180.0 / M_PI, ndt.hasConverged(), moving_backward);
            accepted = false;
        }
    }
    else {
        transform_matrix = ndt_result;
        transform_matrix(2, 3) = init_guess(2, 3);  // restore original z
        reject_count = 0;
    }




    // Constrain pitch/roll to zero, keep only yaw
    float yaw_angle = atan2(transform_matrix(1, 0), transform_matrix(0, 0));
    float prev_yaw = atan2(init_guess(1, 0), init_guess(0, 0));

    // --- 直線偵測 ---
    yaw_history.push_back(prev_yaw * 180.0 / M_PI);
    if ((int)yaw_history.size() > STRAIGHT_HISTORY_SIZE)
        yaw_history.erase(yaw_history.begin());

    is_straight_line = false;
    if ((int)yaw_history.size() >= STRAIGHT_HISTORY_SIZE) {
        // 計算最近 N 幀 yaw 的標準差（degree）
        float sum = 0, sum2 = 0;
        // 先用相鄰差值來避免角度 wrap-around 問題
        std::vector<float> unwrapped(yaw_history.size());
        unwrapped[0] = yaw_history[0];
        for (size_t i = 1; i < yaw_history.size(); i++) {
            float diff = yaw_history[i] - yaw_history[i - 1];
            while (diff > 180.0) diff -= 360.0;
            while (diff < -180.0) diff += 360.0;
            unwrapped[i] = unwrapped[i - 1] + diff;
        }
        for (size_t i = 0; i < unwrapped.size(); i++) {
            sum += unwrapped[i];
            sum2 += unwrapped[i] * unwrapped[i];
        }
        float n = (float)unwrapped.size();
        float variance = sum2 / n - (sum / n) * (sum / n);
        float stddev = sqrt(fmax(variance, 0.0));
        if (stddev < STRAIGHT_YAW_THRESHOLD)
            is_straight_line = true;
    }

    // --- 自適應 EMA smoothing ---
    // 分解位移為縱向（沿車頭）和橫向（垂直車頭）
    float ndt_dx = transform_matrix(0, 3) - init_guess(0, 3);
    float ndt_dy = transform_matrix(1, 3) - init_guess(1, 3);
    float heading_cos = cos(prev_yaw);
    float heading_sin = sin(prev_yaw);
    float longitudinal = ndt_dx * heading_cos + ndt_dy * heading_sin;   // 沿車頭分量
    float lateral = -ndt_dx * heading_sin + ndt_dy * heading_cos;       // 垂直車頭分量

    float alpha_longitudinal = 0.7;     // 縱向永遠正常
    float alpha_lateral = 0.7;          // 橫向預設正常
    float alpha_yaw = 0.7;             // yaw 預設正常

    if (is_straight_line) {
        alpha_lateral = 0.15;           // 直線時大幅壓低橫向權重
        alpha_yaw = 0.15;              // 直線時大幅壓低 yaw 權重
        // 分數閥值 1.5 和 迭代次數 30 可依據你的 bag log 微調
        if (ndt_score > 1.5 || ndt.getFinalNumIteration() > 30) {
            ROS_WARN("Straight line & Poor NDT detected (Score: %.2f). Relying on prediction.", ndt_score);
            alpha_longitudinal = 0.4;  // 縱向也稍微降低，相信推算
            alpha_lateral = 0.02;      // 幾乎不採用 NDT 的橫向偏移修正
            alpha_yaw = 0.02;          // 幾乎不採用 NDT 的旋轉修正
        }
        else {
            // 在彎道或特徵豐富區域，若分數很好，完全信任 NDT
            if (ndt_score < 0.5) {
                alpha_lateral = 0.9;
                alpha_yaw = 0.9;
        }
        }
    }

    // 用各自的 alpha 平滑
    float smooth_longitudinal = alpha_longitudinal * longitudinal;
    float smooth_lateral = alpha_lateral * lateral;

    // 重新組合回 x, y
    float smooth_x = init_guess(0, 3) + smooth_longitudinal * heading_cos - smooth_lateral * heading_sin;
    float smooth_y = init_guess(1, 3) + smooth_longitudinal * heading_sin + smooth_lateral * heading_cos;

    // Smooth yaw with angle wrapping
    float dyaw_smooth = yaw_angle - prev_yaw;
    while (dyaw_smooth > M_PI) dyaw_smooth -= 2 * M_PI;
    while (dyaw_smooth < -M_PI) dyaw_smooth += 2 * M_PI;
    float smooth_yaw = prev_yaw + alpha_yaw * dyaw_smooth;

    transform_matrix(0, 3) = smooth_x;
    transform_matrix(1, 3) = smooth_y;
    transform_matrix(0, 0) = cos(smooth_yaw);  transform_matrix(0, 1) = -sin(smooth_yaw); transform_matrix(0, 2) = 0;
    transform_matrix(1, 0) = sin(smooth_yaw);  transform_matrix(1, 1) =  cos(smooth_yaw); transform_matrix(1, 2) = 0;
    transform_matrix(2, 0) = 0;                transform_matrix(2, 1) = 0;                transform_matrix(2, 2) = 1;


    // --- 更新歷史位姿，供下一幀 CVM 預測使用 ---
    prev_prev_pose = prev_pose;
    prev_pose = transform_matrix;
    if (pose_history_count < 2) pose_history_count++;

    // Debug info
    frame_count++;
    float x = transform_matrix(0, 3);
    float y = transform_matrix(1, 3);
    float yaw = atan2(transform_matrix(1, 0), transform_matrix(0, 0)) * 180.0 / M_PI;
    ROS_INFO("Frame %d | converged: %d | score: %.4f | iter: %d | x: %.2f y: %.2f yaw: %.1f | radar pts: %lu->%lu | submap pts: %lu",
             frame_count,
             ndt.hasConverged(),
             ndt.getFitnessScore(),
             (int)ndt.getFinalNumIteration(),
             x, y, yaw,
             radar_pc->size(),
             radar_pc_filtered->size(),
             submap_pc->size());

//-------------------------------------------------------

    radar_pose_publisher(transform_matrix(0, 3), transform_matrix(1, 3),  transform_matrix(2, 3), atan2(transform_matrix(1, 0), transform_matrix(0, 0)));
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