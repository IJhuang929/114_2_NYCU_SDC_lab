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

// modify IO path
string map_source_path = "/root/catkin_ws/src/localization/data/Track1/";
string map_save_path = "/root/catkin_ws/src/localization/data/Track1/";

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
// ----------------------------------------------------------------------
    printf("Original map points: %lu\n", cloud->size());

    // 1. PassThrough: filter z range to remove outliers above/below ground
    pcl::PointCloud<pcl::PointXYZI>::Ptr z_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-20.0, 30.0);
    pass.filter(*z_filtered);
    printf("After z filter: %lu\n", z_filtered->size());

    // 2. Statistical Outlier Removal: remove noise points
    pcl::PointCloud<pcl::PointXYZI>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(z_filtered);
    sor.setMeanK(20);              // consider 20 nearest neighbors
    sor.setStddevMulThresh(1.5);   // points beyond 1.5x stddev are removed
    sor.filter(*sor_filtered);
    printf("After SOR filter: %lu\n", sor_filtered->size());

    // 3. VoxelGrid: downsample to reduce density
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(sor_filtered);
    voxel.setLeafSize(0.3, 0.3, 0.3);
    voxel.filter(*filtered_cloud);
    printf("After voxel downsample: %lu\n", filtered_cloud->size());
//-----------------------------------------------------------------------
    filtered_cloud->width = 1;
    filtered_cloud->height = filtered_cloud->points.size();

    pcl::io::savePCDFile<pcl::PointXYZI>(map_save_path + "test.pcd", *filtered_cloud);
}

int main(int argc, char** argv)
{
    map_source_path = "/root/catkin_ws/src/localization/data/Track1/";
    map_save_path = "/root/catkin_ws/src/localization/data/Track1/";
    process_map();
    return 0;
}