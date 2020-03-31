/**
 * @file GraphManager.cpp
 * @brief Factor graph manager for iSAM2 SLAM backend.
 * @date Mar 30, 2020
 * @author tonio terán (teran@mit.edu)
 * @author aldo terán (aldot@kth.se)
 */

#include "GraphManager.h"

namespace fsslam {

GraphManager::GraphManager() { SetupNoiseModels(); }

GraphManager::GraphManager(const double prior_pos_stddev,
                           const double prior_rot_stddev,
                           const double imu_accel_stddev,
                           const double imu_omega_stddev)
    : prior_pos_stddev_(prior_pos_stddev),
      prior_rot_stddev_(prior_rot_stddev),
      imu_accel_stddev_(imu_accel_stddev),
      imu_omega_stddev_(imu_omega_stddev) {
  SetupNoiseModels();
  SetupiSAM();
}

GraphManager::~GraphManager() {}

void GraphManager::SetupNoiseModels() {
  // Setup the prior noise model.
  gtsam::Vector prior_sigmas(6);
  prior_sigmas << prior_pos_stddev_, prior_pos_stddev_, prior_pos_stddev_,
      prior_rot_stddev_, prior_rot_stddev_, prior_rot_stddev_;
  prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);

  // TODO (tonioteran): setup the IMU noise model.
}

void GraphManager::SetupiSAM() {
  gtsam::ISAM2Params params;
  // TODO(tonioteran) using default parameters for the moment. If we want to
  // configure later on, we need to add the desired params to the ctor.
  isam2_ = gtsam::ISAM2(params);
}

void GraphManager::InitFactorGraph(const gtsam::Pose3 &pose) {
  gtsam::Symbol pose_id = gtsam::Symbol('x', cur_pose_idx_);
  // Add prior factor to graph.
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(pose_id, pose,
                                                          prior_noise_);
  // Add initial estimates.
  initial_estimates_.insert(pose_id, pose);
  // Save initial pose as our current pose estimate.
  cur_pose_estimate_ = pose;
  dead_reckoning_ = pose;
  // Toggle flag for factor graph initialization.
  fg_init_ = true;
}

}  // namespace fsslam
