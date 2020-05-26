#!/usr/bin/python2.7
"""
Class to handle the Two-View Bundle Adjustment algorithm
explained in the paper "Degeneracy-Aware Imaging Sonar
Simultaneous Localization and Mapping" by Westman and Kaess.
"""
import numpy as np
from scipy.linalg import sqrtm#, ldl

import tf
import rospy
import ros_numpy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseWithCovariance, PoseStamped
from sensor_msgs.msg import PointCloud2

# for debugging
import test_utils as utils

__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

class BundleAdjuster:
    """
    TBA
    """
    def __init__(self, verbose=True, test=False, debug=False,
                 benchmark=False, iters=10, svd_thresh=50):
        # Flags
        self.is_test = test
        self.is_debug = debug
        self.is_benchmark = benchmark
        self.is_singular = False
        self.verbose = verbose

        # GN params
        # TODO(aldoteran): add this to the config file
        self.iters = iters
        self.svd_threshold = svd_thresh
        self.phi_range = np.arange(-0.54977, 0.54977, 0.017453)
        # self.phi_range = np.arange(-0.1047, 0.1047, 0.0087)
        self.pointcloud = [[],[],[]]
        #### PUBLISHERS ####
        self.pose_pub = rospy.Publisher('/bundle_adjustment/sonar_pose',
                                        PoseStamped,
                                        queue_size=1)
        self.pose_constraint_pub = rospy.Publisher('/bundle_adjustment/sonar_constraint',
                                                   PoseWithCovarianceStamped,
                                                   queue_size=1)
        self.odom_pub = rospy.Publisher('/bundle_adjustment/sonar_odometry',
                                        Odometry,
                                        queue_size=1)
        self.true_odom_pub = rospy.Publisher('bundle_adjustment/true_sonar_odometry',
                                             Odometry,
                                             queue_size=1)
        self.pc2_pub = rospy.Publisher('bundle_adjustment/landmark_cloud',
                                       PointCloud2,
                                       queue_size=1)
        self.tf_pub = tf.TransformBroadcaster()
        # self.tf_listener = tf.TransformListener(cache_time=rospy.Duration(20))

    def compute_constraint(self, landmarks, theta_stddev=0.01, range_stddev=0.01):
        """
        Compute sonar constraint using landmarks seen in two
        sonar images.

        :param landmarks: list of N landmarks
        :type landmarks: list [Landmark_1,...,Landmark_N]

        :return:
        """
        #TODO: import from config
        sigma = np.diag((theta_stddev**2, range_stddev**2))
        # Compute these only once
        inv_sigma = np.linalg.inv(sigma)
        sqrt_sigma = np.linalg.inv(sqrtm(sigma))

        # init with best info up until this point
        N = len(landmarks)
        Xb = landmarks[0].Xb
        Xa = landmarks[0].Xa
        T_Xb = landmarks[0].rel_pose
        x_init, z_a, z_b = self._init_state(landmarks, Xb, T_Xb, N)

        phis = self._opt_phi_search(x_init, z_b, T_Xb, inv_sigma, N, landmarks)
        # phis = [l.real_phi for l in landmarks]

        # Stop condition
        epsilon = 0.01

        # Gauss-Newton NLS optimization
        for it in range(self.iters):
            # (2) Compute whitened Jacobian A and error vector b
            A = self._get_jacobians(x_init, T_Xb, phis, sqrt_sigma, N)
            b = self._get_error_b(x_init, T_Xb, phis, z_a, z_b, sqrt_sigma, N)

            # (3) SVD of A and thresholding of singular values
            U, S, V = np.linalg.svd(A, full_matrices=False)
            S[S<self.svd_threshold] = 0.0

            # (4) Update initial state
            A_d = U.dot(np.diag(S)).dot(V)
            delta = np.linalg.pinv(A_d).dot(b)
            x_init += delta
            T_Xb = self._update_transform(Xa, x_init)

            # (5) Check if converged
            if np.linalg.norm(delta) < epsilon or it >= self.iters:
                break

        covariance = self._get_sqrt_information(A_d)
        if self.is_singular:
            self.is_singular = False
            return

        # Publish everything
        if self.verbose:
            rospy.loginfo("Resulting relative pose:\n{}".format(T_Xb))
        self._publish_pose_constraint(T_Xb, covariance)
        # self._publish_pose(x_init, sigma, covariance)
        # self._publish_true_odom(Xb)
        # self._publish_pointcloud(x_init, phis)

    def _init_state(self, landmarks, Xb, T_Xb, N):
        x = np.zeros((12+2*N, 1))
        z_a = np.zeros((2*N, 1))
        z_b = np.zeros((2*N, 1))
        # Vectorize Relative transform
        x[0:12,0:] = np.expand_dims(T_Xb[:-1,:].transpose().ravel(),1)
        i = 0
        for l in landmarks:
            x[12+i:12+i+2,:] = l.polar_img1
            z_a[i:i+2,:] = l.polar_img1
            z_b[i:i+2,:] = l.polar_img2
            i += 2

        return (x, z_a, z_b)

    def _opt_phi_search(self, x, z_b, T_Xb, inv_sigma, N, landmarks):
        """
        Search for optimal phi using the list of phis in phi_range.
        """
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]
        phis = []

        for i in range(0,2*N,2):
            best_phi = 0.0
            old_error = 9999
            z_bi = z_b[i:i+2,:]
            polar = x[12+i:12+i+2,:]
            for phi in self.phi_range:
                q_i = rot_Xb.transpose().dot(self._project_coords(polar, phi) - trans_Xb)
                innov = self.cart_to_polar(q_i) - z_bi
                error = innov.transpose().dot(inv_sigma).dot(innov)
                if error < old_error:
                    best_phi = phi
                    old_error = error
            phis.append(best_phi)

        return phis

    def _project_coords(self, polar, phi):
        """
        Returns the projection of the X, Y and Z coordinates
        given the elevation angle phi. Used for the search for
        the optimal phi. Initial coords are from img1, i.e., as
        seen from pose X_a.

        :param phi: elevation angle in radians
        :type phi: float

        :return: Z, X and Y coordinates of the landmark
        :rtype: np.array (3,1) [X,Y,Z]
        """
        theta = polar[0,0]
        r = polar[1,0]

        return np.array([[r * np.cos(theta) * np.cos(phi)],
                         [r * np.sin(theta) * np.cos(phi)],
                         [r * np.sin(phi)]])

    def _update_transform(self, Xa, x):
        """
        Update T_Xb
        """
        T_Xb = np.eye(4)
        T_Xb[:-1,-1:] = np.array([[x[9,0]],[x[10,0]],[x[11,0]]])
        T_Xb[:-1,:-1] = np.array([[x[0,0], x[3,0], x[6,0]],
                                  [x[1,0], x[4,0], x[7,0]],
                                  [x[2,0], x[5,0], x[8,0]]])

        return T_Xb

    def _predict_hb(self, T_Xb, cart):
        return T_Xb[:-1,:-1].transpose().dot(cart - T_Xb[:-1, -1:])

    def _get_jacobians(self, x, T_Xb, phis, sqrt_sigma, N):
        R = T_Xb[:-1,:-1]
        t = T_Xb[:-1,-1]

        info_theta = sqrt_sigma[0,0]
        info_range = sqrt_sigma[1,1]
        diag = [info_range if i%2==0 else info_theta for i in range(1,2*N+1,1)]

        H_A = np.hstack((np.zeros((2*N,12)), np.diag(diag)))

        H_B = np.zeros((2*N, 12+2*N))
        for j,i in enumerate(range(0,2*N,2)):
            polar = np.array([[x[12+i,0]],
                              [x[12+i+1,0]],
                              [phis[j]]])
            p = np.squeeze(self.polar_to_cart(polar))
            q = self._predict_hb(T_Xb, p)

            zhat_q = np.array([[-q[1,0]/np.sqrt(q[0,0]**2+q[1,0]**2),
                                q[0,0]/np.sqrt(q[0,0]**2+q[1,0]**2), 0.0],
                            [q[0,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2),
                                q[1,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2),
                                q[2,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2)]])
            q_xb = np.array([[p[0]-t[0], p[1]-t[1], p[2]-t[2], 0, 0, 0, 0, 0, 0,
                              -R[0,0], -R[1,0], -R[2,0]],
                             [0, 0, 0, p[0]-t[0], p[1]-t[1], p[2]-t[2], 0, 0, 0,
                              -R[0,1], -R[1,1], -R[2,1]],
                             [0, 0, 0, 0, 0, 0, p[0]-t[0], p[1]-t[1], p[2]-t[2],
                              -R[0,2], -R[1,2], -R[2,2]]])
            H_B[i:i+2,0:12] = zhat_q.dot(q_xb)

            q_p = R.transpose()
            p_mi = np.array([[-polar[1,0]*np.sin(polar[0,0])*np.cos(polar[2,0]),
                              np.cos(polar[0,0])*np.cos(polar[2,0])],
                            [polar[1,0]*np.cos(polar[0,0])*np.cos(polar[2,0]),
                             np.sin(polar[0,0])*np.cos(polar[2,0])],
                            [0.0, np.sin(polar[2,0])]])
            H_B[i:i+2,12+1*i:12+2+1*i] = zhat_q.dot(q_p).dot(p_mi)
            # Whiten
            H_B[i:i+2,:] = sqrt_sigma.dot(H_B[i:i+2,:])

        return np.vstack((H_A, H_B))

    def _get_error_b(self, x, T_Xb, phis, z_a, z_b, sqrt_sigma, N):
        """
        :param x: state
        :type x: np.array (12+2*N,1)

        :return: predicted range-bearing measurement of landmark from Xa to Xb
        :rtype: np.array (2,1) [theta, range]
        """
        b_a = np.zeros(z_a.shape)
        b_b = np.zeros(z_b.shape)
        rot_Xb = T_Xb[:-1, :-1]
        trans_Xb = T_Xb[:-1, -1:]

        for j,i in enumerate(range(0,2*N,2)):
            polar = np.array([[x[12+i,0]],
                              [x[12+i+1,0]],
                              [phis[j]]])
            b_a[i:i+2,:] = sqrt_sigma.dot(z_a[i:i+2,:] - polar[0:2,:])
            q_i = rot_Xb.transpose().dot(self.polar_to_cart(polar) - trans_Xb)
            b_b[i:i+2,:] = sqrt_sigma.dot(z_b[i:i+2,:] - self.cart_to_polar(q_i))

        return np.vstack((b_a, b_b))

    def _get_sqrt_information(self, A_d):
        """
        Computes the sqrt information matrix as in section V-C in
        the paper.

        :param A_d: SV thresholded A matrix
        :type A_d: np.array (6+2N, 6+2N)

        :return: Square root information matrix R
        :rtype: np.array (6,6)
        """
        Gamma = A_d.transpose().dot(A_d)
        # Shur's complement
        Gamma_11 = Gamma[0:6,0:6]
        Gamma_12 = Gamma[0:6,6:]
        Gamma_21 = Gamma[6:,0:6]
        Gamma_22 = Gamma[6:,6:]
        try:
            Lambda = Gamma_11 - Gamma_12.dot(np.linalg.inv(Gamma_22)).dot(Gamma_21)
            covariance = np.linalg.inv(Lambda)
        except np.linalg.LinAlgError:
            rospy.logerr("Singular matrix encountered when computing covariance, skipping...")
            self.is_singular = True
            return None
        # L, D, P = ldl(Lambda, lower=False)
        # sqrt_inf = sqrtm(D[P,:]).transpose().dot(L[P,:].transpose())

        return covariance

    def _publish_pose_constraint(self, T_Xb, R):
        """
        Publish the sonar pose constraint to add in the factor graph.
        Contains the relative transformation between two sonar images.

        :param T_Xb: Relative transformation matrix
        :type T_Xb: np.array (4,4)

        :param R: square root information matrix
        :type R: np.array (6,6)
        """
        trans_Xb = T_Xb[:-1,-1:]
        quat = tf.transformations.quaternion_from_matrix(T_Xb)
        quat = self.normalize_quat(quat)
        # Publish pose
        sonar_constraint = PoseWithCovarianceStamped()
        sonar_constraint.header.frame_id = "bundle_adjustment/sonar_pose_constraint"
        sonar_constraint.header.stamp = rospy.Time.now()
        sonar_constraint.pose.pose.position.x = trans_Xb[0,0]
        sonar_constraint.pose.pose.position.y = trans_Xb[1,0]
        sonar_constraint.pose.pose.position.z = trans_Xb[2,0]
        sonar_constraint.pose.pose.orientation.x = quat[0]
        sonar_constraint.pose.pose.orientation.y = quat[1]
        sonar_constraint.pose.pose.orientation.z = quat[2]
        sonar_constraint.pose.pose.orientation.w = quat[3]
        sonar_constraint.pose.covariance = R.ravel().tolist()
        self.pose_constraint_pub.publish(sonar_constraint)
        self.tf_pub.sendTransform((trans_Xb[0,0], trans_Xb[1,0], trans_Xb[2,0]),
                                  quat, rospy.Time.now(),
                                  "bundle_adjustment/sonar_pose_constraint",
                                  "slam/dead_reckoning/sonar_pose")

    def _publish_pose(self, x, sigma, R):
        """
        TODO
        """
        trans_Xb = np.array([[x[0,0]],
                             [x[1,0]],
                             [x[2,0]]])
        quat = tf.transformations.quaternion_from_euler(x[5,0],
                                                        x[4,0],
                                                        x[3,0])
        # quat = tf.transformations.quaternion_from_euler(x[3,0],
                                                        # x[4,0],
                                                        # x[5,0])
        quat = self.normalize_quat(quat)
        # Publish pose
        sonar_pose = PoseStamped()
        sonar_pose.header.frame_id = "bundle_adjustment/sonar_estimate"
        sonar_pose.header.stamp = rospy.Time.now()
        sonar_pose.pose.position.x = trans_Xb[0,0]
        sonar_pose.pose.position.y = trans_Xb[1,0]
        sonar_pose.pose.position.z = trans_Xb[2,0]
        sonar_pose.pose.orientation.x = quat[0]
        sonar_pose.pose.orientation.y = quat[1]
        sonar_pose.pose.orientation.z = quat[2]
        sonar_pose.pose.orientation.w = quat[3]
        # publish tf for debugging
        self.tf_pub.sendTransform((trans_Xb[0,0], trans_Xb[1,0], trans_Xb[2,0]),
                                  quat, rospy.Time.now(),
                                  "bundle_adjustment/sonar_estimate",
                                  "/rexrov/forward_sonar_optical_frame")
        self.pose_pub.publish(sonar_pose)
        # Publish odometry
        sonar_pose = PoseWithCovariance()
        sonar_pose.pose.position.x = trans_Xb[0,0]
        sonar_pose.pose.position.y = trans_Xb[1,0]
        sonar_pose.pose.position.z = trans_Xb[2,0]
        sonar_pose.pose.orientation.x = quat[0]
        sonar_pose.pose.orientation.y = quat[1]
        sonar_pose.pose.orientation.z = quat[2]
        sonar_pose.pose.orientation.w = quat[3]
        sonar_pose.covariance = R.ravel().tolist()
        sonar_odom = Odometry()
        sonar_odom.header.frame_id = "/rexrov/forward_sonar_optical_frame"
        sonar_odom.child_frame_id = "bundle_adjustment/sonar_estimate"
        sonar_odom.header.stamp = rospy.Time.now()
        sonar_odom.pose = sonar_pose
        self.odom_pub.publish(sonar_odom)

    def _publish_true_odom(self, Xb):
        quat = tf.transformations.quaternion_from_matrix(Xb)
        trans_Xb = Xb[:-1,-1:]
        # Publish odometry
        sonar_pose = PoseWithCovariance()
        sonar_pose.pose.position.x = trans_Xb[0,0]
        sonar_pose.pose.position.y = trans_Xb[1,0]
        sonar_pose.pose.position.z = trans_Xb[2,0]
        sonar_pose.pose.orientation.x = quat[0]
        sonar_pose.pose.orientation.y = quat[1]
        sonar_pose.pose.orientation.z = quat[2]
        sonar_pose.pose.orientation.w = quat[3]
        sonar_odom = Odometry()
        sonar_odom.header.frame_id = "/world"
        sonar_odom.child_frame_id = "bundle_adjustment/sonar_estimate"
        sonar_odom.header.stamp = rospy.Time.now()
        sonar_odom.pose = sonar_pose
        self.true_odom_pub.publish(sonar_odom)

    def _publish_pointcloud(self, state, phis):
        """
        Update and publish the optimized posityion of the landmarks
        as a PointCloud2 message.

        :param state: Optimized state
        :type state: np.array (6+2N,1)
        """
        p = [1,2,0]
        pointcloud = [[], [], []]
        # Append points to cloud
        for j,i in enumerate(range(6,state.shape[0],2)):
            cart = self.polar_to_cart(np.array([[state[i,0]],
                                                [state[i+1,0]],
                                                [phis[j]]]))
            cart = cart[p]
            pointcloud[0].append(cart[0])
            pointcloud[1].append(cart[1])
            pointcloud[2].append(cart[2])
        # Create record array
        pointcloud = np.squeeze(np.asarray(pointcloud))
        cloud_rarray = np.rec.array([(pointcloud[0,:]),
                                     (pointcloud[1,:]),
                                     (pointcloud[2,:])],
                                    dtype=[('x', 'f4'),
                                           ('y', 'f4'),
                                           ('z', 'f4')])
        # Compose message and publish
        pc2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_rarray,
                                                              rospy.Time.now(),
                                                              '/rexrov/forward_sonar_optical_frame')
        self.pc2_pub.publish(pc2_msg)

    def normalize_quat(self,quat):
        quat = np.asarray(quat)
        length = np.linalg.norm(quat)

        return (quat/length).tolist()

    def polar_to_cart(self, polar):
        return np.array([[polar[1,0] * np.cos(polar[0,0]) * np.cos(polar[2,0])],
                         [polar[1,0] * np.sin(polar[0,0]) * np.cos(polar[2,0])],
                         [polar[1,0] * np.sin(polar[2,0])]])

    def cart_to_polar(self, cart):
        return np.array([[np.arctan2(cart[1,0],cart[0,0])],
                         [np.sqrt(cart[0,0]**2 + cart[1,0]**2 + cart[2,0]**2)]])














