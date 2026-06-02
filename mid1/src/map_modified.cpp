#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>

using namespace std;

string map_source_path;
string map_save_path;

// Parameters (all from params.yaml)
double voxel_leaf_size;
double z_min, z_max;
int sor_mean_k;
double sor_std_dev;

// Radar intensity augmentation parameters
string radar_pcd_folder;   // preprocessed radar PCD with intensity
string radar_t_list;       // radar timestamp list
string radar_result_file;  // localization result (3x4 row-major poses)
double augment_search_radius;  // radius for averaging radar intensity onto map

// ============================================================
//  Radar Intensity Augmentation
//  Projects radar scans onto the map using known poses,
//  computes average compensated intensity per map point.
//  Result: map intensity field = average radar return strength.
// ============================================================
bool augment_map_with_radar_intensity(
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud)
{
    // Load radar timestamps
    ifstream t_in(radar_t_list);
    if (!t_in.is_open()) {
        ROS_ERROR("Cannot open radar_t_list: %s", radar_t_list.c_str());
        return false;
    }
    vector<string> timestamps;
    string line;
    while (getline(t_in, line))
        if (!line.empty()) timestamps.push_back(line);

    // Load poses from result file (3x4 row-major per frame)
    ifstream pose_in(radar_result_file);
    if (!pose_in.is_open()) {
        ROS_ERROR("Cannot open result file: %s", radar_result_file.c_str());
        return false;
    }
    vector<Eigen::Matrix4f> poses;
    while (getline(pose_in, line)) {
        istringstream iss(line);
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        // Each line: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
        vector<float> vals;
        float v;
        while (iss >> v) vals.push_back(v);
        if (vals.size() >= 12) {
            m(0,0)=vals[0]; m(0,1)=vals[1]; m(0,2)=vals[2];  m(0,3)=vals[3];
            m(1,0)=vals[4]; m(1,1)=vals[5]; m(1,2)=vals[6];  m(1,3)=vals[7];
            m(2,0)=vals[8]; m(2,1)=vals[9]; m(2,2)=vals[10]; m(2,3)=vals[11];
            poses.push_back(m);
        }
    }

    if (timestamps.size() != poses.size()) {
        ROS_ERROR("Timestamp count (%lu) != pose count (%lu)",
                  timestamps.size(), poses.size());
        return false;
    }
    ROS_INFO("Augmenting map with %lu radar frames", timestamps.size());

    // Build KD-tree on map points (XYZ only)
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(map_cloud);

    // Accumulate radar intensity per map point
    vector<double> intensity_sum(map_cloud->size(), 0.0);
    vector<int>    intensity_cnt(map_cloud->size(), 0);

    for (size_t i = 0; i < timestamps.size(); ++i) {
        string pcd_path = radar_pcd_folder + "/" + timestamps[i] + ".pcd";
        pcl::PointCloud<pcl::PointXYZI>::Ptr radar(new pcl::PointCloud<pcl::PointXYZI>);
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *radar) == -1)
            continue;

        // Transform radar to map frame
        pcl::PointCloud<pcl::PointXYZI>::Ptr radar_map(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::transformPointCloud(*radar, *radar_map, poses[i]);

        // For each transformed radar point, find nearby map points
        vector<int> indices;
        vector<float> dists;
        for (const auto& rpt : radar_map->points) {
            pcl::PointXYZI search_pt;
            search_pt.x = rpt.x; search_pt.y = rpt.y; search_pt.z = rpt.z;

            if (kdtree.radiusSearch(search_pt, augment_search_radius, indices, dists) > 0) {
                for (int idx : indices) {
                    intensity_sum[idx] += rpt.intensity;
                    intensity_cnt[idx]++;
                }
            }
        }

        if ((i + 1) % 200 == 0)
            ROS_INFO("  Processed %lu / %lu frames", i + 1, timestamps.size());
    }

    // Write average intensity back to map
    int augmented = 0;
    for (size_t i = 0; i < map_cloud->size(); ++i) {
        if (intensity_cnt[i] > 0) {
            map_cloud->points[i].intensity =
                static_cast<float>(intensity_sum[i] / intensity_cnt[i]);
            augmented++;
        }
        // Points with no radar coverage keep their original lidar intensity
    }

    ROS_INFO("Augmented %d / %lu map points with radar intensity",
             augmented, map_cloud->size());
    return true;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr process_map()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    // Step 1: Load PCD
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_source_path, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file map pcd file\n");
        return nullptr;
    }
    ROS_INFO("Original map size: %zu", cloud->size());

    // Step 2: PassThrough z filter
    pcl::PointCloud<pcl::PointXYZI>::Ptr z_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*z_filtered);
    ROS_INFO("After z filter (z: [%.1f, %.1f]): %zu", z_min, z_max, z_filtered->size());

    // Step 3: Statistical Outlier Removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(z_filtered);
    sor.setMeanK(sor_mean_k);
    sor.setStddevMulThresh(sor_std_dev);
    sor.filter(*sor_filtered);
    ROS_INFO("After SOR (k=%d, std=%.1f): %zu", sor_mean_k, sor_std_dev, sor_filtered->size());

    // Step 4: VoxelGrid downsample
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(sor_filtered);
    float leaf = static_cast<float>(voxel_leaf_size);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*filtered_cloud);
    ROS_INFO("After voxel downsample (leaf=%.2f): %zu", voxel_leaf_size, filtered_cloud->size());

    // Step 5: Save result
    filtered_cloud->width = 1;
    filtered_cloud->height = filtered_cloud->points.size();

    pcl::io::savePCDFile<pcl::PointXYZI>(map_save_path, *filtered_cloud);
    ROS_INFO("Saved preprocessed map to: %s (%zu points)", map_save_path.c_str(), filtered_cloud->size());

    return filtered_cloud;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_map_preprocessor");
    ros::NodeHandle nh("~");

    // Read all parameters from param server (loaded via params.yaml)
    nh.param<string>("lidar_map_pcd_path", map_source_path, "");
    nh.param<string>("preprocessed_map_save_path", map_save_path, "");
    nh.param<double>("lidar_voxel_size", voxel_leaf_size, 0.3);
    nh.param<double>("lidar_z_min", z_min, -10.0);
    nh.param<double>("lidar_z_max", z_max, 40.0);
    nh.param<int>("sor_mean_k", sor_mean_k, 20);
    nh.param<double>("sor_std_dev", sor_std_dev, 1.5);

    // Radar augmentation parameters
    bool augment_radar = false;
    nh.param<bool>("augment_radar_intensity", augment_radar, false);
    nh.param<string>("preprocessed_radar_pcd_folder", radar_pcd_folder, "");
    nh.param<string>("radar_t_list", radar_t_list, "");
    nh.param<string>("radar_result_file", radar_result_file, "");
    nh.param<double>("augment_search_radius", augment_search_radius, 1.0);

    ROS_INFO("=== Map Preprocessor Parameters ===");
    ROS_INFO("  source: %s", map_source_path.c_str());
    ROS_INFO("  save:   %s", map_save_path.c_str());
    ROS_INFO("  z: [%.1f, %.1f], voxel: %.2f, SOR(k=%d, std=%.1f)",
             z_min, z_max, voxel_leaf_size, sor_mean_k, sor_std_dev);
    if (augment_radar)
        ROS_INFO("  Radar augmentation ON (radius=%.2f)", augment_search_radius);

    // Process map (geometry filtering + downsampling)
    pcl::PointCloud<pcl::PointXYZI>::Ptr processed_map = process_map();
    if (!processed_map)
    {
        ROS_ERROR("Map processing failed. Exiting.");
        return -1;
    }

    // Optional: augment map with radar intensity
    if (augment_radar) {
        ROS_INFO("=== Augmenting map with radar intensity ===");
        if (augment_map_with_radar_intensity(processed_map)) {
            // Re-save with radar intensity encoded
            string augmented_path = map_save_path;
            // Insert "_radar" before .pcd extension
            size_t dot = augmented_path.rfind(".pcd");
            if (dot != string::npos)
                augmented_path.insert(dot, "_radar");
            else
                augmented_path += "_radar.pcd";

            pcl::io::savePCDFile<pcl::PointXYZI>(augmented_path, *processed_map);
            ROS_INFO("Saved radar-augmented map: %s (%zu pts)",
                     augmented_path.c_str(), processed_map->size());
        }
    }

    // Publish map for rviz visualization
    ros::Publisher pub_lidar_map = nh.advertise<sensor_msgs::PointCloud2>("/lidar_map", 1, true);

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*processed_map, map_msg);
    map_msg.header.frame_id = "map";

    pub_lidar_map.publish(map_msg);
    ROS_INFO("Published /lidar_map (%zu points)", processed_map->size());

    ros::spin();
    return 0;
}
