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
  // Setup the sonar two view noise model.
  gtsam::Vector sonar_sigmas(6);
  sonar_sigmas << sonar_pos_stddev_, sonar_pos_stddev_, sonar_pos_stddev_,
      sonar_rot_stddev_, sonar_rot_stddev_, sonar_rot_stddev_;
  sonar_noise_ = gtsam::noiseModel::Diagonal::Sigmas(sonar_sigmas);
  // Setup the sonar two view noise model.
  gtsam::Vector sonar_ex_sigmas(6);
  sonar_ex_sigmas << sonar_ex_pos_stddev_, sonar_ex_pos_stddev_, sonar_ex_pos_stddev_,
      sonar_ex_rot_stddev_, sonar_ex_rot_stddev_, sonar_ex_rot_stddev_;
  sonar_extrinsics_noise_ = gtsam::noiseModel::Diagonal::Sigmas(sonar_ex_sigmas);
  // Setup velocity noise model.
  gtsam::Vector vel_sigmas(3);
  vel_sigmas << vel_stddev_, vel_stddev_, vel_stddev_;
  vel_noise_ = gtsam::noiseModel::Diagonal::Sigmas(vel_sigmas);
  // Fixed IMU noise
  gtsam::Vector imu_sigmas(6);
  imu_sigmas << imu_accel_stddev_, imu_accel_stddev_, imu_accel_stddev_,
      imu_omega_stddev_, imu_omega_stddev_, imu_omega_stddev_;
  imu_noise_ = gtsam::noiseModel::Diagonal::Sigmas(imu_sigmas);
  // Prior IMU bias.
  gtsam::Vector imu_bias_sigmas(6);
  imu_bias_sigmas << imu_accel_bias_stddev_, imu_accel_bias_stddev_,
                     imu_accel_bias_stddev_, imu_omega_bias_stddev_,
                     imu_omega_bias_stddev_, imu_omega_bias_stddev_;
  imu_bias_noise_ = gtsam::noiseModel::Diagonal::Sigmas(imu_bias_sigmas);
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
      gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(0.00);

  // TODO(tonioteran): pull out to params yaml.
  // This parameters are explained here: https://gtsam.org/doxygen/a03439.html
  p->accelerometerCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(imu_accel_stddev_, 2);
  p->gyroscopeCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(imu_omega_stddev_, 2);
  p->biasAccCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(imu_accel_bias_stddev_, 2);
  p->biasOmegaCovariance =
      gtsam::Matrix33::Identity(3, 3) * pow(imu_omega_bias_stddev_, 2);
  p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * 1e-8;
  p->biasAccOmegaInt = gtsam::Matrix::Identity(6, 6) * 1e-5;

  // Instantiate both odometers.
  odometer_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
          p, cur_imu_bias_);
  accumulator_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
          p, cur_imu_bias_);
}

void GraphManager::InitFactorGraph(const gtsam::Pose3 &pose) {
  gtsam::Symbol pose_id = gtsam::Symbol('x', cur_odom_pose_idx_);
  gtsam::Symbol bias_id = gtsam::Symbol('b', cur_bias_idx_);
  gtsam::Symbol vel_id = gtsam::Symbol('v', cur_vel_idx_);
  gtsam::Symbol sonar_id = gtsam::Symbol('s', cur_pose_idx_);

  //gtsam::Pose3 pose;
  gtsam::Pose3 sonar_init_pose = pose * sonar_extrinsics_;

  // Add prior factor to graph.
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(pose_id, pose,
                                                          prior_noise_);
  // Add prior sonar factor to graph.
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(sonar_id,
                                                          sonar_init_pose,
                                                          prior_noise_);
  // Add prior velocity to graph.
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(vel_id,
                                                           cur_vel_estimate_,
                                                           vel_noise_);
  // Add prior IMU bias factor to graph.
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          bias_id, cur_imu_bias_, imu_bias_noise_);

  // Add initial estimates.
  initial_estimates_.insert(pose_id, pose);
  initial_estimates_.insert(sonar_id, sonar_init_pose);
  initial_estimates_.insert(vel_id, cur_vel_estimate_);
  initial_estimates_.insert(bias_id, cur_imu_bias_);
  // Save initial pose as our current estimate.
  cur_pose_estimate_ = pose;
  dead_reckoning_ = pose;
  // Save initial origin for IMU preintegration
  cur_state_estimate_ = gtsam::NavState(cur_pose_estimate_, cur_vel_estimate_);
  // Toggle flag for factor graph initialization.
  fg_init_ = true;
}

void GraphManager::AddImuMeasurement(const Eigen::Vector3d &accel,
                                     const Eigen::Vector3d &omega,
                                     const double dt) {

  // Preintegrate on both the `odometer_` and the `accumulator_`.
  odometer_->integrateMeasurement(accel, omega, dt);
  accumulator_->integrateMeasurement(accel, omega, dt);

  // Generate the dead-reckoned pose estimate.
  gtsam::NavState dead_reckon = odometer_->predict(cur_state_estimate_, cur_imu_bias_);
  dead_reckoning_ = dead_reckon.pose();
}

void GraphManager::AddImuFactor() {
    // Adds all the relevant factors and updates the Bayes tree
    if(cur_pose_idx_ != cur_sonar_pose_idx_) {
        std::cout << "Node idx mismatch: cur_pose = " << cur_pose_idx_ <<
            "; cur_sonar_pose = " << cur_sonar_pose_idx_ << std::endl;
                 cur_pose_idx_, cur_sonar_pose_idx_;
    } else {
        cur_pose_idx_++;
        gtsam::Key prev_pose = gtsam::Symbol('x', cur_pose_idx_ - 1);
        gtsam::Key cur_pose = gtsam::Symbol('x', cur_pose_idx_);
        gtsam::Key prev_bias = gtsam::Symbol('b', cur_pose_idx_ - 1);
        gtsam::Key cur_bias = gtsam::Symbol('b', cur_pose_idx_);
        gtsam::Key prev_vel = gtsam::Symbol('v', cur_pose_idx_ - 1);
        gtsam::Key cur_vel = gtsam::Symbol('v', cur_pose_idx_);
        gtsam::Key cur_sonar_pose = gtsam::Symbol('s', cur_pose_idx_);
        // Add IMU factor (already includes bias factor!)
        gtsam::CombinedImuFactor imu_factor(prev_pose, prev_vel, cur_pose, cur_vel,
                                            prev_bias, cur_bias, *odometer_);
        graph_.add(imu_factor);
        // Add sonar extrinsics factor
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(cur_pose,
                                                      cur_sonar_pose,
                                                      sonar_extrinsics_,
                                                      //sonar_extrinsics_noise_));
                                                      sonar_noise_));

        // Update state
        gtsam::NavState predicted_state = odometer_->predict(cur_state_estimate_,
                                                             cur_imu_bias_);
        initial_estimates_.insert(cur_pose, predicted_state.pose());
        initial_estimates_.insert(cur_sonar_pose, predicted_state.pose()*sonar_extrinsics_);
        initial_estimates_.insert(cur_vel, predicted_state.v());
        initial_estimates_.insert(cur_bias, cur_imu_bias_);
    }
}

void GraphManager::AddSonarFactor(gtsam::Pose3 sonar_constraint) {
    cur_sonar_pose_idx_++;
    gtsam::Symbol prev_pose = gtsam::Symbol('s', cur_sonar_pose_idx_ - 1);
    gtsam::Symbol cur_pose = gtsam::Symbol('s', cur_sonar_pose_idx_);
    // Add pose constraint computed by sonar bundle adjustment
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_pose, cur_pose,
                                                  sonar_constraint,
                                                  sonar_noise_));
}

Eigen::Affine3d GraphManager::UpdateiSAM() {
    gtsam::Key cur_pose = gtsam::Symbol('x', cur_pose_idx_);
    gtsam::Key cur_vel = gtsam::Symbol('v', cur_pose_idx_);
    gtsam::Key cur_bias = gtsam::Symbol('b', cur_pose_idx_);
    // Update the Bayes tree
    std::cout << "Factor Graph:" << std::endl;
    graph_.print();

    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimates_);
    gtsam::Values result = optimizer.optimize();

    //isam2_.update(graph_, initial_estimates_);
    //gtsam::Values result = isam2_.calculateEstimate();
    // Update current state and bias with iSAM2 optimized values
    cur_state_estimate_ = gtsam::NavState(result.at<gtsam::Pose3>(cur_pose),
                                          result.at<gtsam::Vector3>(cur_vel));
    std::cout << "Current Estimate:" << std::endl;
    std::cout << cur_state_estimate_ << std::endl;
    cur_imu_bias_ = result.at<gtsam::imuBias::ConstantBias>(cur_bias);

    // Reset odometer with new bias estimate
    odometer_->resetIntegrationAndSetBias(cur_imu_bias_);
    // Reset factor graph
    //graph_ = gtsam::NonlinearFactorGraph();
    // Reset initial estimates
    //initial_estimates_.clear();

    return Eigen::Affine3d(cur_state_estimate_.pose().matrix());
}

void GraphManager::PrintFactorGraph() {
    graph_.print("\n Current Factor Graph:\n");
}

bool GraphManager::isGraphInit() {
    return fg_init_;
}

}  // namespace fsslam
