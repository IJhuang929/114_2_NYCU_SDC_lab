cmake_minimum_required(VERSION 3.0.2)
project(localization)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  tf
  tf_conversions
)


catkin_package()


include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(localization src/localization.cpp)
target_link_libraries(localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(map_modified src/map_modified.cpp)
target_link_libraries(map_modified ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(visualization src/visualization.cpp)
target_link_libraries(visualization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})