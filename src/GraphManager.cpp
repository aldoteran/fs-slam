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
                           const Eigen::Vector3d imu_accel_noise_stddev,
                           const Eigen::Vector3d imu_omega_noise_stddev,
                           const Eigen::Vector3d imu_accel_bias_stddev,
                           const Eigen::Vector3d imu_omega_bias_stddev,
                           const Eigen::Vector3d init_accel_bias,
                           const Eigen::Vector3d init_gyro_bias)
    : prior_pos_stddev_(prior_pos_stddev),
      prior_rot_stddev_(prior_rot_stddev),
      imu_accel_noise_stddev_(imu_accel_noise_stddev),
      imu_omega_noise_stddev_(imu_omega_noise_stddev),
      imu_accel_bias_stddev_(imu_accel_bias_stddev),
      imu_omega_bias_stddev_(imu_omega_bias_stddev),
      init_accel_bias_(init_accel_bias),
      init_gyro_bias_(init_gyro_bias) {
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
    imu_sigmas << imu_accel_noise_stddev_(0), imu_accel_noise_stddev_(1),
                  imu_accel_noise_stddev_(2), imu_omega_noise_stddev_(0),
                  imu_omega_noise_stddev_(1), imu_omega_noise_stddev_(2);
    imu_noise_ = gtsam::noiseModel::Diagonal::Sigmas(imu_sigmas);
    // IMU bias random walk.
    gtsam::Vector imu_bias_sigmas(6);
    imu_bias_sigmas << imu_accel_bias_stddev_(0), imu_accel_bias_stddev_(1),
                       imu_accel_bias_stddev_(2), imu_omega_bias_stddev_(0),
                       imu_omega_bias_stddev_(1), imu_omega_bias_stddev_(2);
    imu_bias_noise_ = gtsam::noiseModel::Diagonal::Sigmas(imu_bias_sigmas);
    // Initial IMU Bias
    cur_imu_bias_ = gtsam::imuBias::ConstantBias(init_accel_bias_,
                                                 init_gyro_bias_);

}

void GraphManager::SetupiSAM() {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    isam2_ = gtsam::ISAM2(params);
}

void GraphManager::SetupOdometers() {
    // NOTE(tonioteran) this chooses a particular direction for the gravity
    // vector, explained here https://gtsam.org/doxygen/a00698_source.html.
    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> p =
      gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(0);

    // TODO(tonioteran): pull out to params yaml.
    // This parameters are explained here: https://gtsam.org/doxygen/a03439.html
    Eigen::DiagonalMatrix<double, 3> accel_cov(pow(imu_accel_noise_stddev_(0), 2),
                                               pow(imu_accel_noise_stddev_(1), 2),
                                               pow(imu_accel_noise_stddev_(2), 2));
    p->accelerometerCovariance = accel_cov;
    Eigen::DiagonalMatrix<double, 3> omega_cov(pow(imu_omega_noise_stddev_(0), 2),
                                               pow(imu_omega_noise_stddev_(1), 2),
                                               pow(imu_omega_noise_stddev_(2), 2));
    p->gyroscopeCovariance = omega_cov;
    Eigen::DiagonalMatrix<double, 3> accel_bias_cov(pow(imu_accel_bias_stddev_(0), 2),
                                                    pow(imu_accel_bias_stddev_(1), 2),
                                                    pow(imu_accel_bias_stddev_(2), 2));
    p->biasAccCovariance = accel_bias_cov;
    Eigen::DiagonalMatrix<double, 3> omega_bias_cov(pow(imu_omega_bias_stddev_(0), 2),
                                                    pow(imu_omega_bias_stddev_(1), 2),
                                                    pow(imu_omega_bias_stddev_(2), 2));
    p->biasOmegaCovariance = omega_bias_cov;
    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * 1e-8;
    p->biasAccOmegaInt = gtsam::Matrix::Identity(6, 6) * 1e-5;

    // Instantiate both odometers.
    odometer_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
          p, cur_imu_bias_);
    accumulator_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
          p, cur_imu_bias_);
}

void GraphManager::InitFactorGraph(const gtsam::Pose3 &pose) {
    // Save initial state
    gtsam::Pose3 sonar_init_pose = initial_pose_ * sonar_extrinsics_;

    std::cout << " ---------- Graph Priors ----------- " << std::endl;
    std::cout << "IMU Initial Pose" << std::endl;
    std::cout << initial_pose_ << std::endl;
    std::cout << "Sonar Initial Pose" << std::endl;
    std::cout << sonar_init_pose << std::endl;

    // Add prior factor to graph.
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), initial_pose_,
                                                          prior_noise_);
    // Add prior sonar factor to graph.
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(S(0),
                                                          sonar_init_pose,
                                                          prior_noise_);
    // Add prior velocity to graph.
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0),
                                                           cur_vel_estimate_,
                                                           vel_noise_);
    // Add prior IMU bias factor to graph.
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          B(0), cur_imu_bias_, imu_bias_noise_);

    // Add initial estimates.
    initial_estimates_.insert(X(0), initial_pose_);
    initial_estimates_.insert(S(0), sonar_init_pose);
    initial_estimates_.insert(V(0), cur_vel_estimate_);
    initial_estimates_.insert(B(0), cur_imu_bias_);
    // Save initial pose as our current estimate.
    cur_pose_estimate_ = initial_pose_;
    dead_reckoning_ = initial_pose_;
    // Save initial origin for IMU preintegration
    cur_state_estimate_ = gtsam::NavState(cur_pose_estimate_, cur_vel_estimate_);

    // Add priors to ISAM2 tree.
    isam2_.update(graph_, initial_estimates_);
    graph_.resize(0);
    initial_estimates_.clear();

    // Toggle flag for factor graph initialization.
    fg_init_ = true;
    cur_frame_ = 1;
}

void GraphManager::AddImuMeasurement(const Eigen::Vector3d &accel,
                                     const Eigen::Vector3d &omega,
                                     const gtsam::Rot3 &orientation,
                                     const double dt) {

    gtsam::Rot3 imu_attitude = orientation;
    if (imu_init_ == false){
        initial_pose_ = gtsam::Pose3(imu_attitude, gtsam::Vector3(0.0,0.0,0.0));
        initial_state_ = gtsam::NavState(initial_pose_, gtsam::Vector3(0.0,0.0,0.0));
        dead_reckoned_state_ = initial_state_;
        imu_init_ == true;
    }

    // Preintegrate on both the `odometer_` and the `accumulator_`.
    odometer_->integrateMeasurement(accel, omega, dt);
    accumulator_->integrateMeasurement(accel, omega, dt);

    // Generate the dead-reckoned pose estimate.
    gtsam::imuBias::ConstantBias imu_bias;
    dead_reckoned_state_ = accumulator_->predict(initial_state_, imu_bias);
    dead_reckoning_ = dead_reckoned_state_.pose();
    gtsam::NavState odometry = odometer_->predict(cur_state_estimate_, cur_imu_bias_);
    odometry_ = odometry.pose();
}

Eigen::Affine3d GraphManager::AddFactors(gtsam::Pose3 sonar_constraint,
                                         gtsam::Matrix66 R) {
    // This function will be triggered by the sonar pose constraint callback
    std::cout << " ---------- Running iSAM for frame: " << cur_frame_;
    std::cout << " ----------" << std::endl;

    // Sonar constraint Covariance
    sonar_covariance_ = gtsam::noiseModel::Gaussian::Covariance(R);

    // Get odometry
    gtsam::NavState prop_state = odometer_->predict(cur_state_estimate_,
                                                    cur_imu_bias_);
    gtsam::Pose3 prop_pose = prop_state.pose();
    // Get estimated pose from ISAM2
    gtsam::Pose3 prev_pose = isam2_.calculateEstimate<gtsam::Pose3>(
            X(cur_frame_ - 1));
    // Get estimated velocity from ISAM2
    gtsam::Vector3 vel = isam2_.calculateEstimate<gtsam::Vector3>(
            V(cur_frame_ - 1));

    // Add initial estimate
    initial_estimates_.insert(X(cur_frame_), prev_pose.compose(sonar_constraint));
    initial_estimates_.insert(V(cur_frame_), vel);
    initial_estimates_.insert(B(cur_frame_), cur_imu_bias_);
    initial_estimates_.insert(S(cur_frame_), prev_pose.compose(sonar_constraint)
                                             * sonar_extrinsics_);

    std::cout << "Preintegrated Estimate" << std::endl;
    std::cout << prop_state << std::endl;
    std::cout << "ISAM2 Previous Estimate" << std::endl;
    std::cout << prev_pose << std::endl;
    std::cout << "Composed Pose" << std::endl;
    std::cout << prev_pose.compose(sonar_constraint) << std::endl;
    std::cout << "Estimated Sonar Pose" << std::endl;
    std::cout << prev_pose.compose(sonar_constraint)*sonar_extrinsics_ << std::endl;
    std::cout << "Estimated IMU Bias" << std::endl;
    std::cout << cur_imu_bias_ << std::endl;

    // Add IMU factor
    gtsam::CombinedImuFactor imu_factor(X(cur_frame_ - 1), V(cur_frame_ -1),
                                        X(cur_frame_), V(cur_frame_),
                                        B(cur_frame_ - 1), B(cur_frame_),
                                        *odometer_);
    graph_.add(imu_factor);
    // Add sonar extrinsics factor
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(cur_frame_),
                                                  S(cur_frame_),
                                                  sonar_extrinsics_,
                                                  sonar_extrinsics_noise_));
    // Add sonar pose constraint
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(S(cur_frame_ - 1),
                                                  S(cur_frame_),
                                                  sonar_constraint,
                                                  sonar_covariance_));

    // Update ISAM2
    isam2_.update(graph_, initial_estimates_);

    for(int k = 0; k < 3; k++)
        isam2_.update();
    gtsam::Values results = isam2_.calculateEstimate();

    // Update current state and bias with iSAM2 optimized values
    cur_state_estimate_ = gtsam::NavState(results.at<gtsam::Pose3>(X(cur_frame_)),
                                          results.at<gtsam::Vector3>(V(cur_frame_)));
    cur_imu_bias_ = results.at<gtsam::imuBias::ConstantBias>(B(cur_frame_));

    // Reset odometer with new bias estimate
    odometer_->resetIntegration();
    std::cout << "ISAM2 Optimized Estimate" << std::endl;
    std::cout << cur_state_estimate_ << std::endl;

    // Reset factor graph
    graph_.resize(0);
    // Reset initial estimates
    initial_estimates_.clear();
    // Done with current frame, increase counter
    cur_frame_++;

    // Return latest state estimate for ROS
    return Eigen::Affine3d(cur_state_estimate_.pose().matrix());
}

}  // namespace fsslam
