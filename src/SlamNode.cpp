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
  // Subscribe to the sonar bundle adjustment pose constraint topic
  sonar_pose_sub_ =
      nh_.subscribe(sonar_pose_topic_, 1000, &SlamNode::SonarPoseCallback, this);
  // Subscriber for the pose constraint covariance
  sonar_covariance_sub_ =
      nh_.subscribe(sonar_covariance_topic_, 1000, &SlamNode::SonarCovarianceCallback, this);

  // Setup the static SLAM path publisher for dead reckoned trajectory.
  dead_reckoning_pub_ =
      nh_.advertise<nav_msgs::Path>(dead_reckoning_topic_, 1000);
  dead_reckoning_path_.header.frame_id = dead_reckoning_frame_id_;
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

    ROS_INFO("Found TF for origin.");
    std::cout << origin_.matrix() << std::endl;

    ROS_INFO("Initializing Factor Graph with found origin.");
    gtsam::Rot3 rot{origin_.rotation().matrix()};
    gtsam::Point3 trans{origin_.translation()};
    gtsam::Pose3 state_origin(rot, trans);
    gm_->InitFactorGraph(state_origin);
    // Publish first optimized pose and TF
    PublishOptimizedPath(Eigen::Affine3d(state_origin.matrix()));
  }
  catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
    ROS_WARN("TF for origin not found, defaulting to zero.");
    gm_->InitFactorGraph(gtsam::Pose3());
    // Publish first optimized pose and TF
    PublishOptimizedPath(Eigen::Affine3d());
  }
}

void SlamNode::ImuMeasCallback(const sensor_msgs::Imu &msg) {
  Eigen::Vector3d accel = {msg.linear_acceleration.x, msg.linear_acceleration.y,
                           msg.linear_acceleration.z};
  Eigen::Vector3d omega = {msg.angular_velocity.x, msg.angular_velocity.y,
                           msg.angular_velocity.z};

  gm_->AddImuMeasurement(accel, omega, imu_dt_);

  PublishDeadReckonPath();
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
        //std::cout << "Rotation:" << rotation.matrix() << std::endl;
        //std::cout << "Translation:" << position << std::endl;
        // Update iSAM and get optimized pose
        Eigen::Affine3d opt_pose = gm_->AddFactors(
                gtsam::Pose3(rotation, position), covariance);
        // Publish the pose
        PublishOptimizedPath(opt_pose);
        PublishSonarTF();
        PublishTruePath();
    }
}

//void SlamNode::SonarPoseCallback(
        //const geometry_msgs::PoseWithCovarianceStamped &msg) {
    //if(gm_->isGraphInit()) {
        //// Add Sonar pose constraint
        //gtsam::Point3 position(msg.pose.pose.position.x,
                               //msg.pose.pose.position.y,
                               //msg.pose.pose.position.z);
        //gtsam::Rot3 rotation(msg.pose.pose.orientation.w,
                                //msg.pose.pose.orientation.x,
                                //msg.pose.pose.orientation.y,
                                //msg.pose.pose.orientation.z);
        //pose_constraint_ = gtsam::Pose3(rotation, position);
        //ROS_WARN("Processed sonar constraint.");
    //}
//}

void SlamNode::SonarCovarianceCallback(const std_msgs::Float32MultiArray &msg){
        auto cov = msg.data;
        gtsam::Matrix covariance(12,12);
        for (int i = 0; i < covariance.rows(); ++i)
            for (int j = 0; j < covariance.cols(); ++j)
                covariance(i, j) = cov[i + j*covariance.rows()];

        //ROS_WARN("Adding sonar pose constraint to graph.");
        //Eigen::Affine3d opt_pose = gm_->AddFactors(pose_constraint_, covariance);
        //PublishOptimizedPath(opt_pose);
        //PublishSonarTF();
        //PublishTruePath();
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

void SlamNode::PublishTruePath() {
    tf::StampedTransform transform;
    ros::Time now = ros::Time::now();
    Eigen::Affine3d pose;
    // get TF from world to base link
    tf_listener_.waitForTransform(map_frame_id_, imu_frame_id_,
                                now, ros::Duration(5.0));
    tf_listener_.lookupTransform(map_frame_id_, "rexrov/base_link",
                                now, transform);
    tf::transformTFToEigen(transform, pose);
    geometry_msgs::PoseStamped pose_msg = TransformToPose(pose);
    // Publish path
    pose_msg.header.frame_id = true_pose_frame_id_;
    true_path_.poses.push_back(pose_msg);
    // Publish updated path onto topic.
    true_pose_pub_.publish(true_path_);
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

  fsslam::SlamNode slam_node;
  ros::Rate loop_rate(slam_node.loop_rate_);

  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
