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
  InitState();
}

SlamNode::~SlamNode() {}

void SlamNode::ReadParams() {
  // ROS-related parameters.
  nh_.getParam("node_loop_rate", loop_rate_);

  // Subscriber parameters.
  nh_.getParam("imu_meas_topic", imu_meas_topic_);
  nh_.getParam("sonar_img_topic", sonar_img_topic_);

  // Publisher parameters.
  nh_.getParam("dead_reckoning_topic", dead_reckoning_topic_);
  nh_.getParam("dead_reckoning_odom_topic", dead_reckoning_odom_topic_);
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
  // Subscribe to the IMU sensor measurements.
  sonar_img_sub_ =
      nh_.subscribe(sonar_img_topic_, 1000, &SlamNode::SonarImgCallback, this);
  // Subscribe to the sonar bundle adjustment constraint topic
  sonar_pose_sub_ =
      nh_.subscribe(sonar_pose_topic_, 1000, &SlamNode::SonarPoseCallback, this);

  // Setup the static SLAM path publisher for dead reckoned trajectory.
  dead_reckoning_pub_ =
      nh_.advertise<nav_msgs::Path>(dead_reckoning_topic_, 1000);
  dead_reckoning_path_.header.frame_id = dead_reckoning_frame_id_;
  // Setup the static SLAM path publisher for the optimized trajectory.
  optimized_pose_pub_ =
      nh_.advertise<nav_msgs::Path>(optimized_pose_topic_, 1000);
  optimized_pose_path_.header.frame_id = optimized_pose_frame_id_;

}

void SlamNode::InitGraphManager() {
  gm_ = std::make_unique<GraphManager>(prior_pos_stddev_, prior_rot_stddev_,
                                       imu_accel_stddev_, imu_omega_stddev_);
}

void SlamNode::InitState() {
  ros::Duration(1.0).sleep(); // wait for TF buffer to build up
  tf::StampedTransform transform;
  ros::Time now = ros::Time::now();
  try {
      // Get current TF of the sonar wrt the world
      tf_listener_.waitForTransform(map_frame_id_, imu_frame_id_,
                                    now, ros::Duration(5.0));
      tf_listener_.lookupTransform(map_frame_id_, imu_frame_id_,
                                   now, transform);
      tf::transformTFToEigen(transform, origin_);

      ROS_WARN("Found TF for origin.");
      std::cout << origin_.matrix() << std::endl;

      ROS_WARN("Initializing Factor Graph with origin.");
      gtsam::Rot3 rot{origin_.rotation().matrix()};
      gtsam::Point3 trans{origin_.translation()};
      gtsam::Pose3 state_origin(rot, trans);
      gm_->InitFactorGraph(state_origin);
  }
  catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
    ROS_WARN("TF for origin not found, defaulting to zero.");
  }
}

void SlamNode::ImuMeasCallback(const sensor_msgs::Imu &msg) {
  Eigen::Vector3d accel = {msg.linear_acceleration.x, msg.linear_acceleration.y,
                           msg.linear_acceleration.y};
  Eigen::Vector3d omega = {msg.angular_velocity.x, msg.angular_velocity.y,
                           msg.angular_velocity.y};

  gm_->AddImuMeasurement(accel, omega, imu_dt_);

  PublishDeadReckonPath();
}

void SlamNode::SonarImgCallback(const sensor_msgs::Image &msg) {
    if(gm_->isGraphInit()) {
        node_count_++;
        // Add IMU odometry pose constraint
        ROS_WARN("Adding new pose to graph.");
        gm_->AddImuFactor();
    }
}

void SlamNode::SonarPoseCallback(
        const geometry_msgs::PoseStamped &msg) {
    if(gm_->isGraphInit()) {
        // Add Sonar pose constraint
        gtsam::Point3 position(msg.pose.position.x,
                               msg.pose.position.y,
                               msg.pose.position.z);
        gtsam::Rot3 rotation = {msg.pose.orientation.x,
                                msg.pose.orientation.y,
                                msg.pose.orientation.z,
                                msg.pose.orientation.w};
        ROS_WARN("Adding sonar pose constraint to graph.");
        std::cout << "Rotation:" << rotation.matrix() << std::endl;
        std::cout << "Translation:" << position << std::endl;
        gm_->AddSonarFactor(gtsam::Pose3(rotation, position));

        // Update iSAM and get optimized pose
        Eigen::Affine3d opt_pose = gm_->UpdateiSAM();
        // Publish the pose
        PublishOptimizedPath(opt_pose);
    }
}

geometry_msgs::PoseStamped SlamNode::TransformToPose(
    const Eigen::Affine3d &tfm) {
  geometry_msgs::PoseStamped pose;
  tf::poseEigenToMsg(tfm, pose.pose);
  return pose;
}

void SlamNode::PublishOptimizedPath(Eigen::Affine3d pose) {
    // Update optimized path with latest pose.
    geometry_msgs::PoseStamped pose_msg = TransformToPose(pose);
    pose_msg.header.frame_id = optimized_pose_frame_id_;
    optimized_pose_path_.poses.push_back(pose_msg);
    // Publish updated path onto topic.
    optimized_pose_pub_.publish(optimized_pose_path_);
    // Publish the updated TF
    tf::Transform transform;
    tf::poseEigenToTF(pose, transform);
    br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                      optimized_pose_frame_id_, "slam/sonar_optimized_link"));
}

void SlamNode::PublishDeadReckonPath() {
  // Update our path with the latest pose.
  Eigen::Affine3d pose = gm_->GetDeadReckoning();
  geometry_msgs::PoseStamped pose_msg = TransformToPose(pose);
  pose_msg.header.frame_id = dead_reckoning_frame_id_;
  dead_reckoning_path_.poses.push_back(pose_msg);
  // Publish updated path onto topic.
  dead_reckoning_pub_.publish(dead_reckoning_path_);
  // Publish the updated pose on the TF server
  tf::Transform transform_;
  tf::poseEigenToTF(pose, transform_);
  br_.sendTransform(tf::StampedTransform(transform_, ros::Time::now(),
                    dead_reckoning_frame_id_, "imu/odom"));
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
