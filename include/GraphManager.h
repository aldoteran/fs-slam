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
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>

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

  //! Get the latest dead reckoned pose estimate.
  inline const Eigen::Affine3d GetDeadReckoning() const {
    return Eigen::Affine3d(dead_reckoning_.matrix());
  }

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
  //! Standard deviation parameter for IMU's acceleration components.
  const double imu_accel_stddev_ = 0.001;  // [???], uninformed guess.
  //! Standard deviation parameter for IMU's angular velocity components.
  const double imu_omega_stddev_ = 0.001;  // [???], uninformed guess.

  //! Setup iSAM's optimization and inference parameters.
  void SetupiSAM();
  //! Bayes tree for incremental smoothing and mapping.
  gtsam::ISAM2 isam2_;

  //! Values to capture initial estimates for the new nodes.
  gtsam::Values initial_estimates_;
  //! Holds the full system's state estimate (poses + landmarks).
  gtsam::Values cur_sys_estimate_;
  //! Contains the current pose estimate (initially at origin).
  gtsam::Pose3 cur_pose_estimate_ = gtsam::Pose3();
  //! Contains the hitherto dead reckoning pose estimate (initially at origin).
  gtsam::Pose3 dead_reckoning_ = gtsam::Pose3();

  //! Initializes the factor graph with first pose estimate and prior.
  void InitFactorGraph(const gtsam::Pose3 &pose);
  //! Factor graph for new observations.
  gtsam::NonlinearFactorGraph graph_;
  //! Flag for checking whether the factor graph has been initialized.
  bool fg_init_ = false;

  //! Variable for keeping track of the number of poses in the graph. Counter
  //! should be incremented at the beginning of an `AddPose` function.
  int cur_pose_idx_ = 0;
};

}  // namespace fsslam

#endif  // FS_SLAM_GRAPH_MANAGER_H_
