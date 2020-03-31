/**
 * @file SLAMNode.h
 * @brief ROS node for SLAM.
 * @date Mar 30, 2020
 * @author tonio terán (teran@mit.edu)
 * @author aldo terán (aldot@kth.se)
 */

#ifndef FS_SLAM_SLAM_NODE_H_
#define FS_SLAM_SLAM_NODE_H_

#include "GraphManager.h"

#include <ros/console.h>
#include <ros/ros.h>

#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>

#include <iostream>

namespace fsslam {

//! Ros node that encapsulates the SLAM system.
/*!
  Handles the message-triggered callbacks by instantiating a `GraphManager`
  object onto which to redirect the messages. Information fusion and inference
  is carried out by iSAM2 and GTSAM in the background.
 */
class SlamNode {
 public:
  SlamNode();
  virtual ~SlamNode();

  //! Desired frequency at which to operate the ROS node [hz].
  int loop_rate_ = 50;

 private:
  //! Reads in the yaml file with configuration parameters.
  void ReadParams();
  //! Setup Ros subscribers, publishers, services, etc.
  void SetupRos();
  //! Initializes the backend SLAM object using defined parameters.
  void InitGraphManager();

  //! Ros node handle.
  ros::NodeHandle nh_;

  //! Subscriber for the IMU information.
  ros::Subscriber imu_meas_sub_;
  //! Topic in which the IMU measurements are being received.
  std::string imu_meas_topic_ = "/rexrov/imu";
  //! Callback for the IMU measurements subscriber.
  void ImuMeasCallback(const sensor_msgs::Imu &msg);

  //! Publisher for SLAM dead reckoning.
  ros::Publisher dead_reckoning_pub_;
  nav_msgs::Path dead_reckoning_path_;
  //! Default name for static SLAM trajectory path.
  std::string dead_reckoning_topic_ = "/slam/dead_reckoning";
  //! Default name for frame in which the static SLAM path is expressed.
  std::string dead_reckoning_frame_id_ = "map";

  //! Standard deviation parameter for prior's translation component [m].
  double prior_pos_stddev_;
  //! Standard deviation parameter for prior's rotation component [rad].
  double prior_rot_stddev_;
  //! Standard deviation parameter for IMU's acceleration component [???].
  double imu_accel_stddev_;
  //! Standard deviation parameter for IMU's angluar velocity component [???].
  double imu_omega_stddev_;

  //! Factor graph manager for SAM and inference.
  std::unique_ptr<GraphManager> gm_ = nullptr;
};

}  // namespace fsslam

#endif  // FS_SLAM_SLAM_NODE_H_
