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
  //InitState();
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

  // Imu noise model parameters.
  nh_.getParam("accel_noise_x", imu_accel_noise_stddev_(0));
  nh_.getParam("accel_noise_y", imu_accel_noise_stddev_(1));
  nh_.getParam("accel_noise_z", imu_accel_noise_stddev_(2));
  nh_.getParam("accel_bias_x", imu_accel_bias_stddev_(0));
  nh_.getParam("accel_bias_y", imu_accel_bias_stddev_(1));
  nh_.getParam("accel_bias_z", imu_accel_bias_stddev_(2));
  nh_.getParam("gyro_noise_x", imu_omega_noise_stddev_(0));
  nh_.getParam("gyro_noise_y", imu_omega_noise_stddev_(1));
  nh_.getParam("gyro_noise_z", imu_omega_noise_stddev_(2));
  nh_.getParam("gyro_bias_x", imu_omega_bias_stddev_(0));
  nh_.getParam("gyro_bias_y", imu_omega_bias_stddev_(1));
  nh_.getParam("gyro_bias_z", imu_omega_bias_stddev_(2));
  // Previously estimated initial biases
  nh_.getParam("init_accel_bias_x", init_accel_bias_(0));
  nh_.getParam("init_accel_bias_y", init_accel_bias_(1));
  nh_.getParam("init_accel_bias_z", init_accel_bias_(2));
  nh_.getParam("init_gyro_bias_x", init_gyro_bias_(0));
  nh_.getParam("init_gyro_bias_y", init_gyro_bias_(1));
  nh_.getParam("init_gyro_bias_z", init_gyro_bias_(2));


  // Sensor parameters.
  double imu_frequency = 25;  // [Hz]
  nh_.getParam("imu_frequency", imu_frequency);
  imu_dt_ = 1.0 / imu_frequency;
}

void SlamNode::SetupRos() {
  // Subscribe to the IMU sensor measurements.
  imu_meas_sub_ =
      nh_.subscribe(imu_meas_topic_, 1000, &SlamNode::ImuMeasCallback, this);
  // Subscribe to the sonar bundle adjustment pose constraint topic
  sonar_pose_sub_ =
      nh_.subscribe(sonar_pose_topic_, 1000, &SlamNode::SonarPoseCallback, this);

  // Setup the static SLAM path publisher for dead reckoned trajectory.
  dead_reckoning_pub_ =
      nh_.advertise<nav_msgs::Path>(dead_reckoning_topic_, 1000);
  dead_reckoning_path_.header.frame_id = dead_reckoning_frame_id_;
  // Setup the static SLAM path publisher for imu odom trajectory.
  imu_odom_pub_ =
      nh_.advertise<nav_msgs::Path>(imu_odom_topic_, 1000);
  imu_odom_path_.header.frame_id = dead_reckoning_frame_id_;
  // Setup the static SLAM path publisher for the optimized trajectory.
  optimized_pose_pub_ =
      nh_.advertise<nav_msgs::Path>(optimized_pose_topic_, 1000);
  optimized_pose_path_.header.frame_id = optimized_pose_frame_id_;
  // Setup the static SLAM path publisher for the true trajectory.
  true_pose_pub_ =
      nh_.advertise<nav_msgs::Path>(true_pose_topic_, 1000);
  true_path_.header.frame_id = true_pose_frame_id_;


}

void SlamNode::InitGraphManager() {
  gm_ = std::make_unique<GraphManager>(prior_pos_stddev_, prior_rot_stddev_,
                                       imu_accel_noise_stddev_, imu_omega_noise_stddev_,
                                       imu_accel_bias_stddev_, imu_omega_bias_stddev_,
                                       init_accel_bias_, init_gyro_bias_);
}

void SlamNode::InitState() {
    gm_->InitFactorGraph(init_pose_);
    // Publish first optimized pose and TF
    PublishOptimizedPath(Eigen::Affine3d(init_pose_.matrix()));
}

void SlamNode::ImuMeasCallback(const sensor_msgs::Imu &msg) {
    double qx = msg.orientation.x;
    double qy = msg.orientation.y;
    double qz = msg.orientation.z;
    double qw = msg.orientation.w;
    gtsam::Point3 trans(0.0,0.0,0.0);
    gtsam::Rot3 rot = gtsam::Rot3(qw, qx, qy, qz);

    if (is_init_ == false){
        init_pose_ = gtsam::Pose3(rot, trans);
        is_init_ = true;
        ROS_WARN("IMU orientation obtained.");
    }

    gtsam::Rot3 orientation(rot);
    Eigen::Vector3d accel = {msg.linear_acceleration.x, msg.linear_acceleration.y,
                             msg.linear_acceleration.z};
    Eigen::Vector3d omega = {msg.angular_velocity.x, msg.angular_velocity.y,
                             msg.angular_velocity.z};

    gm_->AddImuMeasurement(accel, omega, orientation, imu_dt_);

    PublishDeadReckonPath();
    PublishImuOdomPath();
}

void SlamNode::SonarPoseCallback(
        const geometry_msgs::PoseWithCovarianceStamped &msg) {
    if(gm_->isGraphInit()) {
        // Add Sonar pose constraint
        gtsam::Point3 position(msg.pose.pose.position.x,
                               msg.pose.pose.position.y,
                               msg.pose.pose.position.z);
        gtsam::Rot3 rotation(msg.pose.pose.orientation.w,
                                msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z);
        auto cov = msg.pose.covariance;
        gtsam::Matrix66 covariance;
        covariance << cov[0], cov[1], cov[2], cov[3], cov[4], cov[5],
                      cov[6], cov[7], cov[8], cov[9], cov[10], cov[11],
                      cov[12], cov[13], cov[14], cov[15], cov[16], cov[17],
                      cov[18], cov[19], cov[20], cov[21], cov[22], cov[23],
                      cov[24], cov[25], cov[26], cov[27], cov[28], cov[29],
                      cov[30], cov[31], cov[32], cov[33], cov[34], cov[35];
        ROS_WARN("Adding sonar pose constraint to graph.");
        std::cout << "Rotation:" << rotation.matrix() << std::endl;
        std::cout << "Translation:" << position << std::endl;
        // Update iSAM and get optimized pose
        Eigen::Affine3d opt_pose = gm_->AddFactors(
                gtsam::Pose3(rotation, position), covariance);
        // Publish the pose
        PublishOptimizedPath(opt_pose);
        PublishSonarTF();
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
                      optimized_pose_frame_id_, "slam/optimized/base_pose"));
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
                    dead_reckoning_frame_id_, "/slam/dead_reckoning/base_pose"));
}

void SlamNode::PublishImuOdomPath() {
    // Update our path with the latest pose.
    Eigen::Affine3d pose = gm_->GetImuOdom();
    geometry_msgs::PoseStamped pose_msg = TransformToPose(pose);
    pose_msg.header.frame_id = dead_reckoning_frame_id_;
    imu_odom_path_.poses.push_back(pose_msg);
    // Publish updated path onto topic.
    imu_odom_pub_.publish(imu_odom_path_);
    // Publish the updated pose on the TF server
    tf::Transform transform_;
    tf::poseEigenToTF(pose, transform_);
    br_.sendTransform(tf::StampedTransform(transform_, ros::Time::now(),
                    dead_reckoning_frame_id_, "/slam/imu_odom/base_pose"));
}

void SlamNode::PublishSonarTF() {
    Eigen::Affine3d pose = gm_->GetSonarExtrinsics();
    // Publish the TF of the sonar
    tf::Transform transform;
    tf::poseEigenToTF(pose, transform);
    br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                      "slam/optimized/base_pose", "slam/optimized/sonar_pose"));
}

}  // namespace fsslam

/* ************************************************************************** */
/* ************************************************************************** */
/* ************************************************************************** */
int main(int argc, char **argv) {
  ros::init(argc, argv, "slam_node");

  bool is_init = false;
  int i = 0;
  fsslam::SlamNode slam_node;
  ros::Rate loop_rate(slam_node.loop_rate_);
  ros::Duration(1.0).sleep();

  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
    i++;
    if (i == 2){
      slam_node.InitState();
      is_init = true;
    }
  }

  return 0;
}
