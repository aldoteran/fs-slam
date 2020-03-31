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
  SetupOdometers();
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

void GraphManager::SetupOdometers() {
  // NOTE(tonioteran) this chooses a particular direction for the gravity
  // vector, explained here https://gtsam.org/doxygen/a00698_source.html.
  boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> p =
      gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);

  // Set the desired parameters. TODO(tonioteran): pull out to params yaml.
  // This parameters are explained here: https://gtsam.org/doxygen/a03439.html
  // Numerical numbers taken from:
  // https://github.com/borglab/gtsam/blob/develop/examples/ImuFactorsExample.cpp
  p->accelerometerCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(0.0003924, 2);
  p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * 1e-8;
  p->gyroscopeCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(0.000205689024915, 2);
  p->biasAccCovariance = gtsam::Matrix33::Identity(3, 3) * pow(0.004905, 2);
  p->biasOmegaCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(0.000001454441043, 2);
  p->biasAccOmegaInt = gtsam::Matrix::Identity(6, 6) * 1e-5;

  // Instantiate both odometers.
  odometer_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(p);
  accumulator_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(p);
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

void GraphManager::AddImuMeasurement(const Eigen::Vector3d &accel,
                                     const Eigen::Vector3d &omega,
                                     const double dt) {
  // Preintegrate on both the `odometer_` and the `accumulator_`.
  odometer_->integrateMeasurement(accel, omega, dt);
  accumulator_->integrateMeasurement(accel, omega, dt);

  // Probably move these to member variables.
  gtsam::NavState state_origin{};
  gtsam::imuBias::ConstantBias bias;

  // Generate the dead-reckoned pose estimate.
  gtsam::NavState dead_reckon = accumulator_->predict(state_origin, bias);
  std::cout << dead_reckon.t() << std::endl;
}

}  // namespace fsslam
