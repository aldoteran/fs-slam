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

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

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
  //! Init flag
  bool is_init_ = false;

  /// Publish hitherto dead reckoned path.
  void PublishDeadReckonPath();
  /// Publish optimized path.
  void PublishOptimizedPath(Eigen::Affine3d pose);
  /// Publish the true path from gazebo.
  void PublishTruePath();
  /// Used for debugging, publishes the sonar TF to the optimized path.
  void PublishSonarTF();

  // Utilities. TODO(tonioteran) Should move to external file.
  geometry_msgs::PoseStamped TransformToPose(const Eigen::Affine3d &tfm);

 private:
  //! Reads in the yaml file with configuration parameters.
  void ReadParams();
  //! Setup Ros subscribers, publishers, services, etc.
  void SetupRos();
  //! Initializes the backend SLAM object using defined parameters.
  void InitGraphManager();
  //! Initializes the state
  void InitState();

  //! Ros node handle.
  ros::NodeHandle nh_;
  //! Ros TF listener.
  tf::TransformListener tf_listener_;
  //! Ros TF broadcaster.
  tf::TransformBroadcaster br_;

  //! Frame ids
  std::string imu_frame_id_ = "rexrov/imu_link";
  std::string map_frame_id_ = "world";

  //! Subscriber for the IMU information.
  ros::Subscriber imu_meas_sub_;
  //! Topic in which the IMU measurements are being received.
  std::string imu_meas_topic_ = "/rexrov/imu";
  //! Callback for the IMU measurements subscriber.
  void ImuMeasCallback(const sensor_msgs::Imu &msg);

  //! Subscriber for the Sonar images.
  ros::Subscriber sonar_img_sub_;
  //! Topic in which the sona images are being received.
  std::string sonar_img_topic_ = "/rexrov/depth/image_raw_raw_sonar";
  //! Callback for the sonar image subscriber.
  void SonarImgCallback(const sensor_msgs::Image &msg);

  //! Subscriber for the sonar pose from the bundle adjustment.
  ros::Subscriber sonar_pose_sub_;
  //! Topic in which the sonar poses are being received.
  std::string sonar_pose_topic_ = "/bundle_adjustment/sonar_constraint";
  //! Callback for the sonar image subscriber.
  void SonarPoseCallback(const geometry_msgs::PoseWithCovarianceStamped &msg);

  //! Initial state
  Eigen::Affine3d origin_;

  //! Publisher for SLAM dead reckoning path.
  ros::Publisher dead_reckoning_pub_;
  nav_msgs::Path dead_reckoning_path_;
  //! Default name for static SLAM trajectory path.
  std::string dead_reckoning_topic_ = "/slam/dead_reckoning/path";
  //! Default name for frame in which the static SLAM path is expressed.
  std::string dead_reckoning_frame_id_ = "world";
  //
  //! Publisher for SLAM dead reckoning path.
  ros::Publisher optimized_pose_pub_;
  nav_msgs::Path optimized_pose_path_;
  //! Default name for static SLAM trajectory path.
  std::string optimized_pose_topic_ = "/slam/optimized/path";
  //! Default name for frame in which the static SLAM path is expressed.
  std::string optimized_pose_frame_id_ = "world";

  //! Publisher for true base path.
  ros::Publisher true_pose_pub_;
  nav_msgs::Path true_path_;
  //! Default name for static SLAM trajectory path.
  std::string true_pose_topic_ = "/slam/true/path";
  //! Default name for frame in which the static SLAM path is expressed.
  std::string true_pose_frame_id_ = "world";

  //! Publisher for SLAM dead reckoning odom.
  ros::Publisher dead_reckoning_odom_pub_;
  nav_msgs::Odometry dead_reckoning_odom_;
  //! Default name for static SLAM trajectory path.
  std::string dead_reckoning_odom_topic_ = "/slam/dead_reckoning/odom";

  //! Standard deviation parameter for prior's translation component [m].
  double prior_pos_stddev_;
  //! Standard deviation parameter for prior's rotation component [rad].
  double prior_rot_stddev_;
  //! Standard deviation parameter for IMU's acceleration component [???].
  Eigen::Vector3d imu_accel_noise_stddev_;
  Eigen::Vector3d imu_accel_bias_stddev_;
  //! Standard deviation parameter for IMU's angluar velocity component [???].
  Eigen::Vector3d imu_omega_noise_stddev_;
  Eigen::Vector3d imu_omega_bias_stddev_;

  /// IMU measurements sampling frequency [s].
  double imu_dt_ = 0.02;  // 50 Hz.

  //! Factor graph manager for SAM and inference.
  std::unique_ptr<GraphManager> gm_ = nullptr;

  //! Counter for the nodes in the Factor Graphs
  int node_count_ = 0;
};

}  // namespace fsslam

#endif  // FS_SLAM_SLAM_NODE_H_
