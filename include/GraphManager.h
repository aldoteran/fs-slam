/**
 * @file GraphManager.h
 * @brief Factor graph manager for iSAM2 SLAM backend.
 * @date Mar 30, 2020
 * @author tonio terán (teran@mit.edu)
 * @author aldo terán (aldot@kth.se)
 */

#ifndef FS_SLAM_GRAPH_MANAGER_H_
#define FS_SLAM_GRAPH_MANAGER_H_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <vector>
#include <iostream>

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::S;

namespace fsslam {

//! Class using GTSAM and iSAM2 to perform SLAM backend tasks.
/*!
  Backend inference class for SLAM using GTSAM/iSAM2. Optimization parameters
  and noise characterisicts are set within `params/slam_params.yaml`.
 */
class GraphManager {
 public:
  // TODO(tonioteran) remove default ctor?
  //! Default, no argument ctor. Uses uninformed guesses as initial parameters.
  GraphManager();
  //! This is the ctor you want, with specifiable parameter values.
  GraphManager(const double prior_pos_stddev, const double prior_rot_stddev,
               const double imu_accel_stddev, const double imu_omega_stddev);
  virtual ~GraphManager();

  /// Predict dead reckoning pose using the accumulated IMU measurements.
  inline const Eigen::Affine3d GetDeadReckoning() const {
    return Eigen::Affine3d(dead_reckoning_.matrix());
  }

  /// Predict the current IMU pose using the corrected IMU measurements.
  inline const Eigen::Affine3d GetImuOdom() const {
    return Eigen::Affine3d(odometry_.matrix());
  }

  /// Return the sonar eextrinsics (for debugging)
  inline const Eigen::Affine3d GetSonarExtrinsics() const {
    return Eigen::Affine3d(sonar_extrinsics_.matrix());
  }

  //! Print current factor graph
  inline void PrintFactorGraph() {
      graph_.print("\n Current Factor Graph:\n");
  }

  //! Return Graph status
  inline bool isGraphInit() {
      return fg_init_;
  }

  /// Preintegrates an IMU measurement onto the `odometer_` and `accumulator_`.
  void AddImuMeasurement(const Eigen::Vector3d &accel,
                         const Eigen::Vector3d &omega,
                         const double dt);
  /// Adds IMU factors and updates Bayes tree with iSAM2.
  Eigen::Affine3d AddFactors(gtsam::Pose3 sonar_constraint,
                             gtsam::Matrix66 R);

  //! Initializes the factor graph with first pose estimate and prior.
  void InitFactorGraph(const gtsam::Pose3 &pose);
  //! Initial pose in the world frame when starting
  gtsam::NavState initial_state_;

 private:
  //! Initialize the noise models using specified parameters.
  void SetupNoiseModels();
  //! Noise model for the prior factor's measurement information. Standard
  //! deviation values with units [m], [m], [m], [rad], [rad], [rad].
  gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
  //! Standard deviation parameter for prior's translation component.
  const double prior_pos_stddev_ = 0.01;  // [m], uninformed guess.
  //! Standard deviation parameter for prior's rotation component.
  const double prior_rot_stddev_ = 0.001;  // [rad], uninformed guess.
  // TODO(aldoteran): This must be computed as in section V-C of the paper.
  //! Noise model for the two view sonar factor's measurement information. Standard
  //! deviation values with units [m], [m], [m], [rad], [rad], [rad].
  gtsam::noiseModel::Diagonal::shared_ptr sonar_noise_;
  gtsam::noiseModel::Gaussian::shared_ptr sonar_covariance_;
  //! Standard deviation parameter for sonar's translation component.
  const double sonar_pos_stddev_ = 0.0001;  // [m], uninformed guess.
  //! Standard deviation parameter for sonars's rotation component.
  const double sonar_rot_stddev_ = 0.0001;  // [rad], uninformed guess.
  //! Very small noise for the sonar extrinsics
  gtsam::noiseModel::Diagonal::shared_ptr sonar_extrinsics_noise_;
  const double sonar_ex_pos_stddev_ = 1e-5;
  const double sonar_ex_rot_stddev_ = 1e-5;

  //! TODO(aldoteran): Noise model for the velocity estimates.
  gtsam::noiseModel::Diagonal::shared_ptr vel_noise_;
  //! Standard deviation for velocity components
  const double vel_stddev_ = 1.0;
  //! Noise model for the IMU measurements
  gtsam::noiseModel::Diagonal::shared_ptr imu_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr imu_bias_noise_;
  //! Standard deviation parameter for IMU's acceleration components.
  const double imu_accel_stddev_ = 4e-3;  // [m/s2], from gazebo.
  //! Standard deviation parameter for IMU's angular velocity components.
  const double imu_omega_stddev_ = 3.39e-4;  // [rad/s], from gazebo.
  //! Stddev for bias parameter for IMU's acceleration components.
  const double imu_accel_bias_stddev_ = 6e-3; // [m/s2], from gazebo.
  //! Stddev for bias parameter for IMU's angular velocity components.
  const double imu_omega_bias_stddev_ = 3.87e-5; // [rad/s], from gazebo.

  //! Setup iSAM's optimization and inference parameters.
  void SetupiSAM();
  //! Bayes tree for incremental smoothing and mapping.
  gtsam::ISAM2 isam2_;

  //! Values to capture initial estimates for the new nodes.
  gtsam::Values initial_estimates_;
  //! Holds the full system's state estimate (poses + landmarks).
  gtsam::Values cur_sys_estimate_;
  //! Holds the most recent pose and velocity for IMU preintegration.
  gtsam::NavState cur_state_estimate_;
  //! Contains the current pose estimate (initially at origin).
  gtsam::Pose3 cur_pose_estimate_ = gtsam::Pose3();
  //! Contains the hitherto dead reckoning pose estimate (initially at origin).
  gtsam::Pose3 dead_reckoning_ = gtsam::Pose3();
  //! Contains the hitherto IMU odometry pose estimate (initially at origin).
  gtsam::Pose3 odometry_ = gtsam::Pose3();
  //! Contains the latest estimate of the velocity.
  gtsam::Vector3 cur_vel_estimate_ = gtsam::Vector3(0.0, 0.0, 0.0);
  ////! Contains the latest IMU bias estimate.
  gtsam::imuBias::ConstantBias cur_imu_bias_;

  //! Sonar extrinsics base (imu) link to sonar optical frame.
  // TODO(aldoteran): move this to config file
  gtsam::Point3 sonar_trans = gtsam::Point3(1.3, 0.0, -0.7);
  gtsam::Rot3 sonar_rot = gtsam::Rot3(0.000778532, 0.977653929,
                                      0.0001674033,-0.210219314);
  gtsam::Pose3 sonar_extrinsics_ = gtsam::Pose3(sonar_rot, sonar_trans);

  /// Odometer for IMU preintegration between keyframes.
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> odometer_ = nullptr;
  /// Accumulator for IMU dead reckoning.
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> accumulator_ =
      nullptr;
  /// Instantiates odometers with custom properties.
  void SetupOdometers();

  //! Factor graph for new observations.
  gtsam::NonlinearFactorGraph graph_;
  //! Flag for checking whether the factor graph has been initialized.
  bool fg_init_ = false;

  //! Variable for keeping track of the number of poses in the graph. Counter
  //! should be incremented at the beginning of an `AddPose` function.
  int cur_pose_idx_ = 0;
  int cur_odom_pose_idx_ = 0;
  int cur_sonar_pose_idx_ = 0;
  int cur_bias_idx_ = 0;
  int cur_vel_idx_ = 0;
  int cur_frame_ = 0;

};

}  // namespace fsslam

#endif  // FS_SLAM_GRAPH_MANAGER_H_
