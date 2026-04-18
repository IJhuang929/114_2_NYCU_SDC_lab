#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <boost/filesystem.hpp>
#include <cmath>
#include <limits>

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
#include <tf2/LinearMath/Quaternion.h>

using namespace std;

ros::Publisher pc_pub;
ros::Publisher submap_pc_pub;
ros::Publisher map_pc_pub;
ros::Publisher radar_pose_pub;
ros::Publisher path_pub;
nav_msgs::Path path;
ofstream file;

// extrinsic calibration from tf_static
Eigen::Matrix4f T_base_to_radar = Eigen::Matrix4f::Identity();
Eigen::Matrix4f T_radar_to_base = Eigen::Matrix4f::Identity();
Eigen::Matrix4f T_map_base = Eigen::Matrix4f::Identity(); // maintained in Base coordinate system for easier IMU integration
std::map<string, float> imu_delta_yaw_map; // store delta_yaw for each timestamp

void callback(float d_yaw);
bool loadInitGuess(const std::string& filename, Eigen::Matrix4f& init_mat);
bool loadImuData(const std::string& filename);
void matrix4fToTfTransform(const Eigen::Matrix4f &matrix, tf::Transform &tf_transform);
void radar_pose_publisher(float x, float y, float z, float yaw);

pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc(new pcl::PointCloud<pcl::PointXYZ>);
pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

bool initialized = false;
Eigen::Matrix4f transform_matrix;   //map to radar (base) transformation matrix to be published and saved
Eigen::Matrix4f initial_guess_from_file = Eigen::Matrix4f::Identity();

// modify IO path
string t_file = "/root/catkin_ws/src/localization/data/Track2/radar_t_list.txt";
string pcd_folder = "/root/catkin_ws/src/localization/data/Track2/preprocessed_radar_pcd_k10_std12_voxel05";
string map_file = "/root/catkin_ws/src/localization/data/Track2/preprocessed_map.pcd";
string save_file = "/root/catkin_ws/src/localization/result/track2_result_04161700.txt";
string init_guess_file = "/root/catkin_ws/src/localization/data/Track2/init_guess.txt";
string imu_file = "/root/catkin_ws/src/localization/data/Track2/imu_aligned.txt";

Eigen::Affine3d radar_ex = Eigen::Affine3d::Identity();

int main(int argc, char** argv) {

    ros::init(argc, argv, "radar_pcd_publisher");
    ros::NodeHandle nh;

    // init static extrinsic matrix (from tf_static)
    T_base_to_radar <<  0.999983,  0.00505, -0.002915,  2.022,
                       -0.005,     0.999843,  0.017014,  0.359,
                        0.003,    -0.016999,  0.999851, -1.15,
                        0.0,       0.0,       0.0,       1.0;

    T_radar_to_base = T_base_to_radar.inverse();

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar", 1);
    submap_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar_submap_pc", 1);
    map_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pc", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/localization_path", 1);

    file.open(save_file);

    tf::TransformBroadcaster br;

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_pc) == -1)
    {
        PCL_ERROR("Couldn't read file pcd\n");
        return -1;
    }

    if (!loadInitGuess(init_guess_file, initial_guess_from_file)) {
        ROS_ERROR("Cannot continue without valid init_guess.txt");
        return -1;
    }

    // --- load IMU data ---
    if (!loadImuData(imu_file)) {
        ROS_WARN("IMU data load failed, will proceed without motion compensation.");
    }

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*map_pc, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    map_pc_pub.publish(map_msg);
    
    ifstream t_in(t_file);
    string line;
    int frame_count = 0;

    while (getline(t_in, line) && ros::ok())
    {
        string pcd_file = pcd_folder + "/" + line + ".pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *radar_pc) == -1)
        {
            PCL_ERROR("Couldn't read file\n");
            continue;
        }

        // --- get time step of delta_yaw ---
        float d_yaw = 0.0;
        if (imu_delta_yaw_map.count(line)) {
            d_yaw = imu_delta_yaw_map[line];
        }

        callback(d_yaw);

        tf::Transform tf_transform1;
        matrix4fToTfTransform(transform_matrix, tf_transform1);
        br.sendTransform(tf::StampedTransform(tf_transform1, ros::Time::now(), "map", "radar"));

        frame_count++;
        if (frame_count % 5 == 0) {
            sensor_msgs::PointCloud2 submap_msg;
            pcl::toROSMsg(*submap_pc, submap_msg);
            submap_msg.header.frame_id = "map";
            submap_msg.header.stamp = ros::Time::now();
            submap_pc_pub.publish(submap_msg);

            sensor_msgs::PointCloud2 radar_msg;
            pcl::toROSMsg(*radar_pc, radar_msg);
            radar_msg.header.frame_id = "radar";
            radar_msg.header.stamp = ros::Time::now();
            pc_pub.publish(radar_msg);
        }

        ros::spinOnce();
    }

    ros::spin();
    return 0;
}

bool loadInitGuess(const std::string& filename, Eigen::Matrix4f& init_mat)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        ROS_ERROR_STREAM("Failed to open init_guess file: " << filename);
        return false;
    }

    std::vector<float> values;
    std::string line;

    while (std::getline(fin, line)) {
        for (char& c : line) {
            if (c == ',' || c == '[' || c == ']' || c == ';') {
                c = ' ';
            }
        }

        std::stringstream ss(line);
        float val;
        while (ss >> val) {
            values.push_back(val);
        }
    }

    fin.close();

    if (values.size() != 16) {
        ROS_ERROR_STREAM("Unexpected init_guess format in " << filename
                         << ", parsed " << values.size() << " numeric values, expected 16.");
        return false;
    }

    int idx = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            init_mat(i, j) = values[idx++];
        }
    }

    ROS_INFO_STREAM("Loaded init_guess from file:\n" << init_mat);
    return true;
}



void callback(float d_yaw){
    Eigen::Matrix4f init_guess_radar;

    if(!initialized){
        init_guess_radar = initial_guess_from_file;
        T_map_base = init_guess_radar * T_radar_to_base; 
        initialized = true;
    }
    else{
        // imu velocity predict
        // rotation increment around z-axis (yaw)
        Eigen::Matrix4f delta_T = Eigen::Matrix4f::Identity();
        delta_T(0, 0) = cos(d_yaw);  delta_T(0, 1) = -sin(d_yaw);
        delta_T(1, 0) = sin(d_yaw);  delta_T(1, 1) = cos(d_yaw);

        // predict current Base pose: previous Base * rotation increment
        T_map_base = T_map_base * delta_T;

        // transform back to Radar coordinate system as NDT's Initial Guess
        // formula: T_map_radar = T_map_base * T_base_radar
        init_guess_radar = T_map_base * T_base_to_radar;
    }
    
    // dynamic crop map into submap
    float center_x = init_guess_radar(0, 3);
    float center_y = init_guess_radar(1, 3);
    float square_size = 250.0;   //  250 * 250 m^2 area around the predicted position

    float half_size = square_size / 2.0;
    Eigen::Vector4f min_pt(center_x - half_size, center_y - half_size,
                           -std::numeric_limits<float>::max(), 1.0);
    Eigen::Vector4f max_pt(center_x + half_size, center_y + half_size,
                           std::numeric_limits<float>::max(), 1.0);

    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
    crop_box_filter.setInputCloud(map_pc);
    crop_box_filter.setMin(min_pt);
    crop_box_filter.setMax(max_pt);
    crop_box_filter.filter(*submap_pc);

    if (radar_pc->empty() || submap_pc->empty()) {
        ROS_WARN("Empty radar/submap cloud after filtering.");
        transform_matrix = init_guess_radar;
        return;
    }
    
    

    // Stage 1: NDT coarse alignment (tuned for accuracy)
    ndt.setInputSource(radar_pc);
    ndt.setInputTarget(submap_pc);
    ndt.setTransformationEpsilon(0.001);
    ndt.setStepSize(0.05);
    ndt.setResolution(2.0);
    ndt.setMaximumIterations(40);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ndt_aligned_pc(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*ndt_aligned_pc, init_guess_radar);

    Eigen::Matrix4f coarse_transform = init_guess_radar;
    double ndt_score = std::numeric_limits<double>::infinity();

    if (ndt.hasConverged()) {
        coarse_transform = ndt.getFinalTransformation();
        ndt_score = ndt.getFitnessScore();
        ROS_INFO_STREAM("NDT coarse converged. score = " << ndt_score);
    } else {
        ROS_WARN("NDT coarse did not converge. Use init_guess for ICP.");

    }

    // Stage 2: ICP refinement (tuned for accuracy)
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(radar_pc);
    icp.setInputTarget(submap_pc);
    icp.setMaxCorrespondenceDistance(0.7);
    icp.setMaximumIterations(20);
    icp.setTransformationEpsilon(1e-5);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setRANSACOutlierRejectionThreshold(0.5);

    pcl::PointCloud<pcl::PointXYZ>::Ptr icp_aligned_pc(new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*icp_aligned_pc, coarse_transform);

    if (icp.hasConverged()) {
        double icp_score = icp.getFitnessScore();
        Eigen::Matrix4f candidate = icp.getFinalTransformation();

        // --- 1. compare NDT and IMU predictions (check NDT reliability) ---
        float ndt_vs_imu_trans = (coarse_transform.block<3,1>(0,3) - init_guess_radar.block<3,1>(0,3)).norm();
        float yaw_imu = std::atan2(init_guess_radar(1, 0), init_guess_radar(0, 0));
        float yaw_ndt = std::atan2(coarse_transform(1, 0), coarse_transform(0, 0));
        float ndt_vs_imu_yaw = std::fabs(yaw_ndt - yaw_imu);
        if (ndt_vs_imu_yaw > M_PI) ndt_vs_imu_yaw = 2 * M_PI - ndt_vs_imu_yaw;

        // --- 2. compare ICP and IMU predictions (check final result reasonableness) ---
        float icp_vs_imu_trans = (candidate.block<3,1>(0,3) - init_guess_radar.block<3,1>(0,3)).norm();
        float yaw_icp = std::atan2(candidate(1, 0), candidate(0, 0));
        float icp_vs_imu_yaw = std::fabs(yaw_icp - yaw_imu);
        if (icp_vs_imu_yaw > M_PI) icp_vs_imu_yaw = 2 * M_PI - icp_vs_imu_yaw;

        // --- 3. compare NDT and ICP (check transformation difference) ---
        float trans_diff_ndt_icp = (candidate.block<3,1>(0,3) - coarse_transform.block<3,1>(0,3)).norm();

        ROS_INFO_STREAM("Validation: NDT_vs_IMU_trans = " << ndt_vs_imu_trans 
                        << ", ICP_vs_IMU_trans = " << icp_vs_imu_trans);

        // --- 4. action logic ---
        // check NDT reliability (compare with IMU prediction, e.g., < 2.0m)
        bool ndt_is_reliable = (ndt_vs_imu_trans < 1.5 && ndt_vs_imu_yaw < 0.2); 

        // check ICP optimization and consistency with prediction (should not exceed 1.0m)
        bool icp_is_better = (icp_score < ndt_score);
        bool icp_is_consistent = (icp_vs_imu_trans < 1.0 && icp_vs_imu_yaw < 0.05);

        if (ndt_is_reliable && icp_is_better && icp_is_consistent) {
            transform_matrix = candidate;
            ROS_INFO("Final Choice: ICP Refined result.");
        } 
        else if (ndt_is_reliable) {
            transform_matrix = coarse_transform;
            ROS_WARN("Final Choice: NDT (ICP rejected by consistency check).");
        }
        else {
            // if NDT far from IMU prediction, fallback to IMU prediction (init_guess) since ICP might have diverged
            transform_matrix = init_guess_radar;
            ROS_ERROR("Final Choice: FORCED IMU PREDICTION (NDT/ICP jumped too far).");
        }

    } else {
        // If ICP did not converge, we have low confidence in the NDT result. We can either fallback to the initial guess or use NDT if it's reasonably close to IMU prediction.
        float ndt_vs_imu_trans = (coarse_transform.block<3,1>(0,3) - init_guess_radar.block<3,1>(0,3)).norm();
        if (ndt_vs_imu_trans < 1.0) {
            transform_matrix = coarse_transform;
        } else {
            transform_matrix = init_guess_radar;
        }
        ROS_WARN("ICP did not converge, validation on NDT.");
    }

    
    //  final Radar pose turn to Base pose
    T_map_base = transform_matrix * T_radar_to_base;

    radar_pose_publisher(transform_matrix(0, 3), transform_matrix(1, 3), transform_matrix(2, 3),
                         atan2(transform_matrix(1, 0), transform_matrix(0, 0)));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) { 
            if(i == 2 && j == 3){
                file << transform_matrix(i, j) << endl;
            }
            else{
                file << transform_matrix(i, j) << " ";
            }
        }
    }
    
    file.flush();
    return;
}
// --- load IMU data ---
bool loadImuData(const std::string& filename) {
    ifstream fin(filename);
    if (!fin.is_open()) return false;
    string line_str, header;
    getline(fin, header); // skip header line
    while (getline(fin, line_str)) {
        stringstream ss(line_str);
        string ts;
        float roll, pitch, yaw, d_yaw;
        ss >> ts >> roll >> pitch >> yaw >> d_yaw;
        imu_delta_yaw_map[ts] = d_yaw;
    }
    fin.close();
    ROS_INFO("IMU data loaded: %zu entries", imu_delta_yaw_map.size());
    return true;
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