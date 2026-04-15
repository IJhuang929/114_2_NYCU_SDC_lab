#include <string>
#include <fstream>
#include <iostream>

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
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

using namespace std;

string map_source_path;
string map_save_path;

// Parameters
double voxel_leaf_size;
double z_min, z_max;
double ransac_distance_threshold;
int sor_mean_k;
double sor_std_dev;

pcl::PointCloud<pcl::PointXYZI>::Ptr process_map()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    // Step 1: Load PCD
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_source_path, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file map pcd file\n");
        return nullptr;
    }
    ROS_INFO("Loaded map: %zu points", cloud->points.size());

    // Step 2: Height filter (remove z < z_min and z > z_max)
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_height_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*cloud_height_filtered);
    ROS_INFO("After height filter (z: [%.1f, %.1f]): %zu points", z_min, z_max, cloud_height_filtered->points.size());

    // Step 3: RANSAC ground removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ransac_distance_threshold);
    seg.setInputCloud(cloud_height_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty())
    {
        ROS_WARN("RANSAC found no plane. Skipping ground removal.");
        *cloud_no_ground = *cloud_height_filtered;
    }
    else
    {
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(cloud_height_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);  // keep non-ground points
        extract.filter(*cloud_no_ground);
        ROS_INFO("After RANSAC ground removal (thresh=%.2f): %zu points (removed %zu ground points)",
                 ransac_distance_threshold, cloud_no_ground->points.size(), inliers->indices.size());
    }

    // Step 4: Voxel Grid Downsample
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(cloud_no_ground);
    float leaf = static_cast<float>(voxel_leaf_size);
    voxel_filter.setLeafSize(leaf, leaf, leaf);
    voxel_filter.filter(*cloud_downsampled);
    ROS_INFO("After voxel downsample (leaf=%.2f): %zu points", voxel_leaf_size, cloud_downsampled->points.size());

    // Step 5: Statistical Outlier Removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cleaned(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud_downsampled);
    sor.setMeanK(sor_mean_k);
    sor.setStddevMulThresh(sor_std_dev);
    sor.filter(*cloud_cleaned);
    ROS_INFO("After SOR (k=%d, std=%.1f): %zu points", sor_mean_k, sor_std_dev, cloud_cleaned->points.size());

    // Step 6: Save result
    cloud_cleaned->width = cloud_cleaned->points.size();
    cloud_cleaned->height = 1;
    cloud_cleaned->is_dense = true;

    pcl::io::savePCDFileBinary<pcl::PointXYZI>(map_save_path, *cloud_cleaned);
    ROS_INFO("Saved preprocessed map to: %s (%zu points)", map_save_path.c_str(), cloud_cleaned->points.size());

    return cloud_cleaned;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_map_preprocessor");
    ros::NodeHandle nh("~");

    // Read parameters from param server (loaded via params.yaml)
    nh.param<string>("lidar_map_pcd_path", map_source_path, "");
    nh.param<string>("preprocessed_map_save_path", map_save_path, "");
    nh.param<double>("lidar_voxel_size", voxel_leaf_size, 0.3);
    nh.param<double>("lidar_z_min", z_min, -1.0);
    nh.param<double>("lidar_z_max", z_max, 4.0);
    nh.param<double>("ransac_distance_threshold", ransac_distance_threshold, 0.2);
    nh.param<int>("sor_mean_k", sor_mean_k, 30);
    nh.param<double>("sor_std_dev", sor_std_dev, 1.0);

    // Process map
    pcl::PointCloud<pcl::PointXYZI>::Ptr processed_map = process_map();
    if (!processed_map)
    {
        ROS_ERROR("Map processing failed. Exiting.");
        return -1;
    }

    // Publish topics
    ros::Publisher pub_lidar_map = nh.advertise<sensor_msgs::PointCloud2>("/lidar_map", 1, true);
    ros::Publisher pub_lidar_map_ndt = nh.advertise<sensor_msgs::PointCloud2>("/lidar_map_for_ndt", 1, true);

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*processed_map, map_msg);
    map_msg.header.frame_id = "map";

    pub_lidar_map.publish(map_msg);
    pub_lidar_map_ndt.publish(map_msg);
    ROS_INFO("Published /lidar_map and /lidar_map_for_ndt (%zu points)", processed_map->points.size());

    ros::spin();
    return 0;
}
