#include <string>
#include <fstream>
#include <iostream>
#include <pcl/filters/voxel_grid.h>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;

string map_source_path;
string map_save_path;

void process_map()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    string map_file = map_source_path + "map.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file map pcd file\n");
        return;
    }

    /*you can filter the map (optional)*/
    // Downsample the point cloud using a VoxelGrid filter
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f); // Adjust leaf size as needed (in meters)
    voxel_filter.filter(*filtered_cloud);


    filtered_cloud->width = 1;
    filtered_cloud->height = filtered_cloud->points.size();

    pcl::io::savePCDFile<pcl::PointXYZI>(map_save_path + "test.pcd", *filtered_cloud);
}

int main(int argc, char** argv)
{
    map_source_path = "/root/catkin_ws/src/localization/data/Track2/";
    map_save_path = "/root/catkin_ws/src/localization/data/Track2/";
    process_map();
    return 0;
}