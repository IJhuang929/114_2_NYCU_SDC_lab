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
#include <pcl/kdtree/kdtree_flann.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <tf/transform_listener.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

#include <Eigen/Eigenvalues>

using namespace std;

ros::Publisher pc_pub;
ros::Publisher submap_pc_pub;
ros::Publisher map_pc_pub;
ros::Publisher radar_pose_pub;
ros::Publisher path_pub;
ros::Publisher fitness_pub;
ros::Publisher degenerate_pub;
ros::Publisher status_pub;
nav_msgs::Path path;
ofstream file;
ofstream diag_file;
int frame_count = 0;

void callback();
void matrix4fToTfTransform(const Eigen::Matrix4f &matrix, tf::Transform &tf_transform);
void radar_pose_publisher(float x, float y, float z, float yaw);

pcl::PointCloud<pcl::PointXYZI>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);

bool initialized = false;
Eigen::Matrix4f transform_matrix;

// Submap cache: avoid re-cropping every frame
Eigen::Vector2f last_submap_center(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
double submap_rebuild_dist = 30.0;  // only rebuild submap when moved > this distance
pcl::KdTreeFLANN<pcl::PointXYZ> submap_kdtree;  // reuse KdTree across frames
bool submap_kdtree_valid = false;

// Adaptive coarse skip: skip coarse NDT when previous frame was good
float prev_fitness_score = std::numeric_limits<float>::max();
double coarse_skip_threshold = 10.0;  // skip coarse if prev score < this

// IO paths (loaded from params.yaml)
string t_file;
string pcd_folder;
string map_file;
string save_dir;
string ego_vel_file;

// === radar_front → base_link 靜態外參 ===
Eigen::Matrix4f T_radar_to_base;

// NDT parameters (defaults match params.yaml, actual values loaded from param server)
double ndt_coarse_voxel;
double ndt_fine_voxel;
int ndt_max_iterations;
double ndt_epsilon;
double fitness_fail_threshold;
double max_jump_distance;
double degenerate_ratio_threshold;
double pose_smooth_alpha;
double submap_size;

// Init guess from params.yaml (row-major 3x4)
vector<double> init_guess_values;

// Ego velocity for current frame (in radar frame)
Eigen::Vector3f current_ego_vel = Eigen::Vector3f::Zero();
double current_dt = 0.0;  // seconds between current and previous frame

// Ego velocity lookup: timestamp_nsec -> [vx, vy, vz]
map<string, Eigen::Vector3f> ego_vel_map;

// ============================================================
//  Load ego_velocity.csv
// ============================================================
void loadEgoVelocity(const string& csv_path)
{
    ifstream fin(csv_path);
    if (!fin.is_open()) {
        ROS_WARN("Cannot open ego_velocity.csv: %s", csv_path.c_str());
        return;
    }
    string header;
    getline(fin, header);  // skip header

    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string ts_str, vx_str, vy_str, vz_str;
        getline(ss, ts_str, ',');
        getline(ss, vx_str, ',');
        getline(ss, vy_str, ',');
        getline(ss, vz_str, ',');
        Eigen::Vector3f vel(stof(vx_str), stof(vy_str), stof(vz_str));
        ego_vel_map[ts_str] = vel;
    }
    fin.close();
    ROS_INFO("Loaded ego velocity for %lu frames", ego_vel_map.size());
}

// ============================================================
//  RCS-weighted point cloud duplication
// ============================================================
pcl::PointCloud<pcl::PointXYZ>::Ptr rcsWeightedCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_xyzi)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    if (cloud_xyzi->empty()) return out;

    int n = cloud_xyzi->points.size();

    // Compute rcs mean and std
    double rcs_sum = 0.0, rcs_sq_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double rcs = cloud_xyzi->points[i].intensity;
        rcs_sum += rcs;
        rcs_sq_sum += rcs * rcs;
    }
    double rcs_mean = rcs_sum / n;
    double rcs_std = sqrt(rcs_sq_sum / n - rcs_mean * rcs_mean);
    if (rcs_std < 1e-6) rcs_std = 1.0;

    out->reserve(n * 2);
    for (int i = 0; i < n; i++) {
        pcl::PointXYZ pt;
        pt.x = cloud_xyzi->points[i].x;
        pt.y = cloud_xyzi->points[i].y;
        pt.z = cloud_xyzi->points[i].z;

        // sigmoid weight normalized to [0.5, 2.0]
        double z_score = (cloud_xyzi->points[i].intensity - rcs_mean) / rcs_std;
        double sigmoid_val = 1.0 / (1.0 + exp(-z_score));
        double weight = 0.5 + sigmoid_val * 1.5;  // maps [0,1] -> [0.5, 2.0]

        out->push_back(pt);
        if (weight > 1.0) {
            out->push_back(pt);  // duplicate high-RCS point
        }
    }
    out->width = out->points.size();
    out->height = 1;
    out->is_dense = true;
    return out;
}

// ============================================================
//  Main callback: Coarse NDT → Fine NDT → Degeneracy → Smooth
// ============================================================
void callback(){
    Eigen::Matrix4f init_guess;

    // Update pose
    if(!initialized){
        if (init_guess_values.size() == 12) {
            init_guess << init_guess_values[0],  init_guess_values[1],  init_guess_values[2],  init_guess_values[3],
                          init_guess_values[4],  init_guess_values[5],  init_guess_values[6],  init_guess_values[7],
                          init_guess_values[8],  init_guess_values[9],  init_guess_values[10], init_guess_values[11],
                          0.0, 0.0, 0.0, 1.0;
        } else {
            // track1
            init_guess << -0.7739014871732518, -0.632632984298013, 0.029188959029395573, -25.31523704528809,
                         0.632582897383501, -0.7743982245509299, -0.012094120483231403, -4.389283657073975,
                         0.030255017582620166, 0.009104778446398086, 0.9995007438318981, -7.73,
                         0.0,      0.0,      0.0,    1.0;
            // track2
            // init_guess << -0.9478046334338953, -0.3188028892577441, 0.005576257002181274, 499.4871520996094,
            //                0.3186884878163892, -0.9466153206969891, 0.04854979253344068, -163.97898864746094,
            //               -0.010199243822116278, 0.047792807227150104, 0.9988051977251651, 7.679019927978516,
            //                0.0,     0.0,     0.0,    1.0;
        }
        initialized = true;
    }
    else{
        init_guess = transform_matrix;

        // Ego velocity prediction: shift init_guess by v_ego * dt
        if (current_dt > 0 && current_ego_vel.norm() > 0.01) {
            // v_ego is in radar/body frame, rotate to map frame
            Eigen::Matrix3f R_map = init_guess.block<3,3>(0,0);
            Eigen::Vector3f vel_map = R_map * current_ego_vel;
            init_guess(0, 3) += vel_map(0) * current_dt;
            init_guess(1, 3) += vel_map(1) * current_dt;
            // Do NOT update z — z is locked to init_guess value (ground vehicle constraint)
        }
    }

    // dynamic crop map into submap — only rebuild when moved far enough
    float center_x = init_guess(0, 3);
    float center_y = init_guess(1, 3);
    Eigen::Vector2f current_center(center_x, center_y);
    float moved = (current_center - last_submap_center).norm();

    if (moved > submap_rebuild_dist || submap_pc->empty()) {
        float half_size = submap_size / 2.0;
        Eigen::Vector4f min_pt(center_x - half_size, center_y - half_size, -std::numeric_limits<float>::max(), 1.0);
        Eigen::Vector4f max_pt(center_x + half_size, center_y + half_size, std::numeric_limits<float>::max(), 1.0);

        pcl::CropBox<pcl::PointXYZ> crop_box_filter;
        crop_box_filter.setInputCloud(map_pc);
        crop_box_filter.setMin(min_pt);
        crop_box_filter.setMax(max_pt);
        crop_box_filter.filter(*submap_pc);

        last_submap_center = current_center;

        // Rebuild KdTree for degeneracy detection
        if (!submap_pc->empty()) {
            submap_kdtree.setInputCloud(submap_pc);
            submap_kdtree_valid = true;
        }
        ROS_INFO("Submap rebuilt: %lu points (moved %.1fm)", submap_pc->points.size(), moved);
    }

    // Convert radar XYZI to XYZ
    pcl::PointCloud<pcl::PointXYZ>::Ptr radar_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*radar_pc, *radar_xyz);

    // ========== Step 2: Coarse NDT (skip if previous frame was good) ==========
    Eigen::Matrix4f coarse_pose;
    bool skipped_coarse = false;

    if (frame_count > 0 && prev_fitness_score < coarse_skip_threshold) {
        // Previous frame converged well + ego velocity prediction → skip coarse
        coarse_pose = init_guess;
        skipped_coarse = true;
    } else {
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_coarse;
        ndt_coarse.setInputSource(radar_xyz);
        ndt_coarse.setInputTarget(submap_pc);
        ndt_coarse.setTransformationEpsilon(0.01);
        ndt_coarse.setStepSize(0.1);
        ndt_coarse.setResolution(ndt_coarse_voxel);
        ndt_coarse.setMaximumIterations(20);  // reduced from 30

        pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_aligned(new pcl::PointCloud<pcl::PointXYZ>);
        ndt_coarse.align(*coarse_aligned, init_guess);
        coarse_pose = ndt_coarse.getFinalTransformation();
    }

    // ========== Step 3: Fine NDT with RCS weighting ==========
    pcl::PointCloud<pcl::PointXYZ>::Ptr weighted_radar = rcsWeightedCloud(radar_pc);

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_fine;
    ndt_fine.setInputSource(weighted_radar);
    ndt_fine.setInputTarget(submap_pc);
    ndt_fine.setTransformationEpsilon(ndt_epsilon);
    ndt_fine.setStepSize(0.1);
    ndt_fine.setResolution(ndt_fine_voxel);
    ndt_fine.setMaximumIterations(30);  // reduced from 50

    pcl::PointCloud<pcl::PointXYZ>::Ptr fine_aligned(new pcl::PointCloud<pcl::PointXYZ>);
    ndt_fine.align(*fine_aligned, coarse_pose);

    Eigen::Matrix4f ndt_result = ndt_fine.getFinalTransformation();
    float score = ndt_fine.getFitnessScore();
    bool converged = ndt_fine.hasConverged();

    // ========== Step 3.5: Lock z, roll, pitch (ground vehicle constraint) ==========
    // Keep z from init_guess (prevent vertical drift)
    ndt_result(2, 3) = init_guess(2, 3);
    // Keep rotation's z-row from init_guess (prevent roll/pitch drift)
    ndt_result(2, 0) = init_guess(2, 0);
    ndt_result(2, 1) = init_guess(2, 1);
    ndt_result(2, 2) = init_guess(2, 2);

    // ========== Step 4: Degeneracy detection ==========
    bool is_degenerate = false;
    float eigen_ratio = 0.0;

    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr src_transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*radar_xyz, *src_transformed, ndt_result);

        // Reuse cached KdTree (rebuilt only when submap changes)
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        int valid_pairs = 0;
        if (submap_kdtree_valid) {
            // Sample every 3rd point for speed
            for (size_t i = 0; i < src_transformed->points.size(); i += 3) {
                vector<int> idx(1);
                vector<float> dist(1);
                if (submap_kdtree.nearestKSearch(src_transformed->points[i], 1, idx, dist) > 0 && dist[0] < 5.0) {
                    Eigen::Vector3f diff(
                        src_transformed->points[i].x - submap_pc->points[idx[0]].x,
                        src_transformed->points[i].y - submap_pc->points[idx[0]].y,
                        src_transformed->points[i].z - submap_pc->points[idx[0]].z);
                    H += diff * diff.transpose();
                    valid_pairs++;
                }
            }
        }
        if (valid_pairs > 0) H /= valid_pairs;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(H);
        Eigen::Vector3f eigenvalues = solver.eigenvalues();  // ascending order

        float lambda_max = eigenvalues(2);
        float lambda_min = eigenvalues(0);

        if (lambda_min > 1e-10)
            eigen_ratio = lambda_max / lambda_min;
        else
            eigen_ratio = 9999.0;

        is_degenerate = (eigen_ratio > degenerate_ratio_threshold);

        if (is_degenerate) {
            // Scale down update in the degenerate direction
            // Log only — do NOT modify ndt_result
            // With sparse radar, degeneracy correction causes more drift than it prevents
            ROS_WARN("Degeneracy detected! ratio=%.1f (no correction applied)", eigen_ratio);
        }
    }

    // Publish degeneracy
    std_msgs::Bool degen_msg;
    degen_msg.data = is_degenerate;
    degenerate_pub.publish(degen_msg);

    // ========== Step 5: Pose validation and smoothing ==========
    string status = "OK";

    if (score > fitness_fail_threshold || !converged) {
        ROS_WARN("NDT failed! score=%.3f converged=%d, keeping previous pose", score, converged);
        ndt_result = (frame_count > 0) ? transform_matrix : init_guess;
        status = "FAILED";
    }
    else {
        // Check jump distance
        Eigen::Vector3f prev_pos = (frame_count > 0) ? transform_matrix.block<3,1>(0,3) : init_guess.block<3,1>(0,3);
        float dx = ndt_result(0, 3) - prev_pos(0);
        float dy = ndt_result(1, 3) - prev_pos(1);
        float jump = sqrt(dx*dx + dy*dy);

        if (frame_count > 0 && jump > max_jump_distance) {
            ROS_WARN("Jump detected! dist=%.2f > %.2f, rejecting", jump, max_jump_distance);
            ndt_result = transform_matrix;
            status = "JUMP_DETECTED";
        }
        else {
            // EMA smoothing (only after first frame)
            if (frame_count > 0) {
                float alpha = pose_smooth_alpha;
                ndt_result.block<3,1>(0,3) = alpha * ndt_result.block<3,1>(0,3)
                                            + (1.0f - alpha) * transform_matrix.block<3,1>(0,3);
            }
            if (is_degenerate) status = "DEGENERATE";
        }
    }

    transform_matrix = ndt_result;
    prev_fitness_score = score;

    // Publish status & fitness
    std_msgs::String status_msg;
    status_msg.data = status;
    status_pub.publish(status_msg);

    std_msgs::Float32 score_msg;
    score_msg.data = score;
    fitness_pub.publish(score_msg);

    // === Log & Diag ===
    frame_count++;
    float x = transform_matrix(0, 3);
    float y = transform_matrix(1, 3);
    float z = transform_matrix(2, 3);
    float yaw = atan2(transform_matrix(1, 0), transform_matrix(0, 0));
    int converged_i = converged ? 1 : 0;

    ROS_INFO("Frame %d | %s | Score: %.3f | x: %.2f y: %.2f z: %.2f | yaw: %.3f | radar: %lu submap: %lu | eigen_ratio: %.1f | coarse: %s",
             frame_count, status.c_str(), score, x, y, z, yaw,
             radar_pc->points.size(), submap_pc->points.size(), eigen_ratio,
             skipped_coarse ? "SKIP" : "RUN");

    diag_file << frame_count << "," << score << "," << x << "," << y << "," << yaw
              << "," << radar_pc->points.size() << "," << submap_pc->points.size()
              << "," << converged_i << "," << status << "," << eigen_ratio << endl;

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

int main(int argc, char** argv) {

    ros::init(argc, argv, "radar_pcd_publisher");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // --- Read IO paths ---
    pnh.param<string>("radar_t_list", t_file, "/root/catkin_ws/src/localization/data/Track1/radar_t_list.txt");
    pnh.param<string>("preprocessed_radar_pcd_folder", pcd_folder, "/root/catkin_ws/src/localization/data/Track1/preprocessed_radar_pcd");
    pnh.param<string>("preprocessed_map_path", map_file, "/root/catkin_ws/src/localization/data/Track1/preprocessed_map.pcd");
    pnh.param<string>("ego_velocity_csv", ego_vel_file, "/root/catkin_ws/src/localization/data/Track1/preprocessed_radar_pcd/ego_velocity.csv");
    pnh.param<string>("result_save_dir", save_dir, "/root/catkin_ws/src/localization/result/");

    // --- Read NDT parameters (all values from params.yaml) ---
    pnh.param<double>("ndt_coarse_voxel", ndt_coarse_voxel, 1.5);
    pnh.param<double>("ndt_fine_voxel", ndt_fine_voxel, 0.5);
    pnh.param<int>("ndt_max_iterations", ndt_max_iterations, 30);
    pnh.param<double>("ndt_epsilon", ndt_epsilon, 0.001);
    pnh.param<double>("fitness_fail_threshold", fitness_fail_threshold, 500.0);
    pnh.param<double>("max_jump_distance", max_jump_distance, 5.0);
    pnh.param<double>("degenerate_ratio_threshold", degenerate_ratio_threshold, 100.0);
    pnh.param<double>("pose_smooth_alpha", pose_smooth_alpha, 0.9);
    pnh.param<double>("submap_size", submap_size, 100.0);
    pnh.param<double>("submap_rebuild_dist", submap_rebuild_dist, 30.0);
    pnh.param<double>("coarse_skip_threshold", coarse_skip_threshold, 10.0);
    pnh.param<vector<double>>("init_guess", init_guess_values, vector<double>());

    // --- Log loaded parameters to verify ---
    ROS_INFO("=== Loaded Parameters ===");
    ROS_INFO("  radar_t_list: %s", t_file.c_str());
    ROS_INFO("  preprocessed_map_path: %s", map_file.c_str());
    ROS_INFO("  ndt_coarse_voxel: %.2f, ndt_fine_voxel: %.2f", ndt_coarse_voxel, ndt_fine_voxel);
    ROS_INFO("  ndt_max_iterations: %d, ndt_epsilon: %.4f", ndt_max_iterations, ndt_epsilon);
    ROS_INFO("  fitness_fail_threshold: %.1f, max_jump_distance: %.1f", fitness_fail_threshold, max_jump_distance);
    ROS_INFO("  pose_smooth_alpha: %.2f, submap_size: %.1f", pose_smooth_alpha, submap_size);
    ROS_INFO("  init_guess size: %lu", init_guess_values.size());
    ROS_INFO("=========================");

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar", 1);
    submap_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar_submap_pc", 1);
    map_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pc", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/localization_path", 1);
    fitness_pub = nh.advertise<std_msgs::Float32>("/localization/fitness_score", 1);
    degenerate_pub = nh.advertise<std_msgs::Bool>("/localization/is_degenerate", 1);
    status_pub = nh.advertise<std_msgs::String>("/localization/status", 1);

    // === result帶時間戳 ===
    boost::filesystem::create_directories(save_dir);
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
    diag_file << "frame,score,x,y,yaw,radar_pts,submap_pts,converged,status,eigen_ratio" << endl;
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

    // === Load ego velocity ===
    loadEgoVelocity(ego_vel_file);

    ifstream t_in(t_file);
    string line;
    string prev_ts = "";

    while(getline(t_in, line) && ros::ok())
    {
        string pcd_file = pcd_folder + "/" + line + ".pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *radar_pc) == -1)
        {
            PCL_ERROR ("Couldn't read file\n");
            continue;
        }

        // Look up ego velocity for this frame
        if (ego_vel_map.count(line)) {
            current_ego_vel = ego_vel_map[line];
        } else {
            current_ego_vel = Eigen::Vector3f::Zero();
        }

        // Compute dt from timestamps
        if (!prev_ts.empty()) {
            double ts_curr = stod(line) / 1e9;
            double ts_prev = stod(prev_ts) / 1e9;
            current_dt = ts_curr - ts_prev;
        } else {
            current_dt = 0.0;
        }
        prev_ts = line;

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
        br.sendTransform(tf::StampedTransform(tf_transform1, ros::Time::now(), "map", "base_link"));

        sensor_msgs::PointCloud2 radar_msg;
        pcl::toROSMsg(*radar_pc, radar_msg);
        radar_msg.header.frame_id = "base_link";
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
