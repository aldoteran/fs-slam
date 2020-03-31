cmake_minimum_required(VERSION 3.10)
project(fs-slam)

set(CMAKE_CXX_STANDARD 17)

# This works for me using a system-wide GTSAM and Eigen installation.
find_package(GTSAM REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

find_package(catkin REQUIRED
  roscpp
  tf
  tf2
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
  )

catkin_package(
  INCLUDE_DIRS
    include
#  LIBRARIES rviz-tools
  CATKIN_DEPENDS
    roscpp
    tf2
    tf2_geometry_msgs
 # DEPENDS system_lib
)

include_directories(
# include
${catkin_INCLUDE_DIRS}
)

# SLAM node.
add_executable(slam_node src/SlamNode.cpp src/GraphManager.cpp)
add_dependencies(slam_node ${catkin_EXPORTED_TARGETS})
target_include_directories(slam_node PUBLIC ./include/)
target_link_libraries(slam_node ${catkin_LIBRARIES} gtsam)

