/**
 * @file SLAMNode.cpp
 * @brief ROS node for SLAM.
 * @date Mar 30, 2020
 * @author tonio terán (teran@mit.edu)
 * @author aldo terán (aldot@kth.se)
 */

#include "SlamNode.h"

namespace fsslam {

SlamNode::SlamNode() {
  ReadParams();
  SetupRos();
  InitGraphManager();
}

SlamNode::~SlamNode() {}

void SlamNode::ReadParams() {
  // ROS-related parameters.
  nh_.getParam("node_loop_rate", loop_rate_);

  // Subscriber parameters.
  nh_.getParam("imu_meas_topic", imu_meas_topic_);

  // Publisher parameters.
  nh_.getParam("dead_reckoning_topic", dead_reckoning_topic_);
  nh_.getParam("dead_reckoning_frame_id", dead_reckoning_frame_id_);

  // Measurement noise model parameters.
  nh_.getParam("prior_pos_stddev", prior_pos_stddev_);
  nh_.getParam("prior_rot_stddev", prior_rot_stddev_);
  nh_.getParam("imu_accel_stddev", imu_accel_stddev_);
  nh_.getParam("imu_omega_stddev", imu_omega_stddev_);

  // Sensor parameters.
  double imu_frequency = 50;  // [Hz]
  nh_.getParam("imu_frequency", imu_frequency);
  imu_dt_ = 1.0 / imu_frequency;
}

void SlamNode::SetupRos() {
  // Subscribe to the IMU sensor measurements.
  imu_meas_sub_ =
      nh_.subscribe(imu_meas_topic_, 1000, &SlamNode::ImuMeasCallback, this);

  // Setup the static SLAM path publisher for dead reckoned trajectory.
  dead_reckoning_pub_ =
      nh_.advertise<nav_msgs::Path>(dead_reckoning_topic_, 1000);
  dead_reckoning_path_.header.frame_id = dead_reckoning_frame_id_;
}

void SlamNode::InitGraphManager() {
  gm_ = std::make_unique<GraphManager>(prior_pos_stddev_, prior_rot_stddev_,
                                       imu_accel_stddev_, imu_omega_stddev_);
}

void SlamNode::ImuMeasCallback(const sensor_msgs::Imu &msg) {
  // TODO (tonioteran): implement me.
  ROS_WARN("ImuMeasCallback() to be implemented!");
  Eigen::Vector3d accel = {msg.linear_acceleration.x, msg.linear_acceleration.y,
                           msg.linear_acceleration.y};
  Eigen::Vector3d omega = {msg.angular_velocity.x, msg.angular_velocity.y,
                           msg.angular_velocity.y};
  gm_->AddImuMeasurement(accel, omega, imu_dt_);

  PublishDeadReckonPath();
}

geometry_msgs::PoseStamped SlamNode::TransformToPose(
    const Eigen::Affine3d &tfm) {
  geometry_msgs::PoseStamped pose;
  tf::poseEigenToMsg(tfm, pose.pose);
  return pose;
}

void SlamNode::PublishDeadReckonPath() {
  // Update our path with the latest pose.
  geometry_msgs::PoseStamped pose = TransformToPose(gm_->GetDeadReckoning());
  pose.header.frame_id = dead_reckoning_frame_id_;
  dead_reckoning_path_.poses.push_back(pose);

  // Publish updated path onto topic.
  dead_reckoning_pub_.publish(dead_reckoning_path_);
}

}  // namespace fsslam

/* ************************************************************************** */
/* ************************************************************************** */
/* ************************************************************************** */
int main(int argc, char **argv) {
  ros::init(argc, argv, "slam_node");

  fsslam::SlamNode slam_node;
  ros::Rate loop_rate(slam_node.loop_rate_);

  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
