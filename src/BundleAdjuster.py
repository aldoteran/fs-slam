#!/usr/bin/python2.7
"""
Class to handle the Two-View Bundle Adjustment algorithm
explained in the paper "Degeneracy-Aware Imaging Sonar
Simultaneous Localization and Mapping" by Westman and Kaess.
"""
import numpy as np
from scipy.linalg import sqrtm, ldl

import tf
import rospy
import ros_numpy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseWithCovariance, PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2

        # PASTE ANYWHERE FOR DEBUGGING
        # import pdb
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # np.set_printoptions(linewidth=250)
        # np.set_printoptions(suppress=True)
        # pdb.set_trace()


__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

class BundleAdjuster:
    """
    TBA
    """
    def __init__(self, iters=10, bearing_stddev=0.05, range_stddev=0.05,
                 vertical_aperture=0.1047, vertical_resolution=0.017, verbose=False):

        # Sonar odometry pose
        self.sonar_pose = None
        self.is_pose_init = False

        # GN params
        self.iters = iters
        self.meas_noise = np.diag((bearing_stddev**2,
                                   range_stddev**2))
        self.phi_range = np.arange(-vertical_aperture,
                                   vertical_aperture,
                                   vertical_resolution)
        # Flags
        self.is_verbose = verbose
        self.is_singular = False
        self.degeneracy_factors = Float32MultiArray()
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
        self.cov_pub = rospy.Publisher('bundle_adjustment/constraint_covariance',
                                        Float32MultiArray,
                                        queue_size=1)
        # for debugging
        self.degeneracy_pub = rospy.Publisher('bundle_adjustment/degeneracy_factors',
                                              Float32MultiArray,
                                              queue_size=1)

        self.tf_listener = tf.TransformListener()
        self.tf_pub = tf.TransformBroadcaster()

    def compute_constraint(self, landmarks, Xa, Xb):
        """
        Compute sonar constraint using landmarks seen in two
        sonar images.

        :param landmarks: list of N landmarks
        :type landmarks: list [Landmark_1,...,Landmark_N]

        :return:
        """
        # Initialize sate
        N = len(landmarks)
        T_Xb = np.linalg.inv(Xa).dot(Xb)
        x_init, z_a, z_b = self._init_state_manifold(landmarks, Xb, T_Xb, N)

        phis = self._opt_phi_search(x_init, z_b, T_Xb, N)
        # phis = [l.real_phi for l in landmarks]

        # Stop condition
        epsilon = 1e-6

        # Gauss-Newton NLS optimization
        for it in range(self.iters):
            # (2) Compute whitened Jacobian A and error vector b
            A = self._get_jacobians_manifold(x_init, T_Xb, phis, N)
            b = self._get_error_b(x_init, T_Xb, phis, z_a, z_b, N)

            # (3) SVD of A and thresholding of singular values
            U, S, V = np.linalg.svd(A, full_matrices=False)
            cond_nums = [np.max(S)/s for s in S]
            thresh = np.argmin((np.asarray(cond_nums)/20.0 - 1)**2)
            S[S<thresh] = 0.0

            # (4) Update initial state
            A_d = U.dot(np.diag(S)).dot(V)
            delta = np.linalg.pinv(A_d).dot(b)
            x_init, T_Xb = self._update_state_manifold(x_init, delta, T_Xb)

            # (5) Check if converged
            if np.linalg.norm(delta) < epsilon or it >= self.iters:
                break

        R_sqrt = self._get_sqrt_information(S, V)
        if self.is_singular:
            rospy.logerr("Singular matrix found while computing0covariance")
            self.is_singular = False
            return

        # Publish everything
        if self.is_verbose:
            rospy.loginfo("Resulting relative pose:\n{}".format(T_Xb))
            euler = tf.transformations.euler_from_matrix(T_Xb)
            rospy.loginfo("RPY:\n{}".format(np.round(euler,4)))
            rospy.loginfo("X: {}".format(T_Xb[0,-1]))
            rospy.loginfo("Y: {}".format(T_Xb[1,-1]))
            rospy.loginfo("Z: {}".format(T_Xb[2,-1]))
        self._publish_pose_constraint(T_Xb, R_sqrt)
        self._publish_pose(R_sqrt, T_Xb)
        self._publish_true_odom(Xb)
        self._publish_pointcloud(x_init, phis)

    def _init_state_manifold(self, landmarks, Xb, T_Xb, N):
        x = np.zeros((6+2*N, 1))
        z_a = np.zeros((2*N, 1))
        z_b = np.zeros((2*N, 1))

        trans_Xb, omega_hat, omega = self.Log_map(T_Xb)

        x[0:6,0:1] = np.array([[trans_Xb[0,-1]],[trans_Xb[1,-1]],[trans_Xb[2,-1]],
                               [omega[0,0]],[omega[1,0]],[omega[2,0]]])

        i = 0
        for l in landmarks:
            x[6+i:6+i+2,:] = l.polar_img1
            z_a[i:i+2,:] = l.polar_img1
            z_b[i:i+2,:] = l.polar_img2
            i += 2

        return (x, z_a, z_b)

    def _opt_phi_search(self, x, z_b, T_Xb, N):
        """
        Search for optimal phi using the list of phis in phi_range.
        """
        inv_sigma = np.linalg.inv(self.meas_noise)
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]
        phis = []

        for i in range(0,2*N,2):
            best_phi = 0.0
            old_error = 9999
            z_bi = z_b[i:i+2,:]
            polar = x[6+i:6+i+2,:]
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

    def Log_map(self, T):
        """
        Compute the Log map of the homogeneous transformation T.
        Return the lie algebra of T.

        :param T: Hom transformation matrix
        :type T: np.array [4x4]

        :return: log map of translation (A^{-1}u) and
                 rotation (log(Rot) = hat{omega}) as a skew symmetric
                 matrix, and the rotation omega
        :rtype: (np.array(3,1), np.array(3,3), np.array(3,1))
        """
        rot = T[:-1,:-1]
        trans = T[:-1,-1:]

        # Get logmap of R to insert in inital state
        log_phi = np.arccos((np.trace(rot)-1)/2)
        # TODO(aldoteran): check if need to wrap to pi
        omega_hat = log_phi/2*np.sin(log_phi) * (rot - rot.transpose())
        omega = np.array([[omega_hat[-1,1]],
                          [omega_hat[0,-1]],
                          [omega_hat[1,0]]])
        omega_norm = np.linalg.norm(omega)

        A_inv = np.eye(3) - (1/2) * omega_hat
        A_inv += omega_hat**2 * (2 * np.sin(omega_norm))-omega_norm*(1+np.cos(omega_norm))\
                 / 2*omega_norm**2*np.sin(omega_norm)
        trans = A_inv.dot(trans)

        return trans, omega_hat, omega

    def _update_state_manifold(self, x, delta_x, T_Xb):
        """
        Update T_Xb on the SE(3) manifold using the exponential map.
        """
        # Update relative transform first
        u = delta_x[0:3,:]
        omega = delta_x[3:6,:]
        omega_norm = np.linalg.norm(omega)
        omega_hat = np.array([[0, -omega[2,0], omega[1,0]],
                              [omega[2,0], 0, -omega[0,0]],
                              [-omega[1,0], omega[0,0], 0]])

        A = np.eye(3)
        A += omega_hat * (1-np.cos(omega_norm))/omega_norm**2
        A += omega_hat**2 * (omega_norm - np.sin(omega_norm))/omega_norm**3

        exp_omega = np.eye(3)
        exp_omega += omega_hat * np.sin(omega_norm)/omega_norm
        exp_omega += omega_hat**2 * (1-np.cos(omega_norm))/omega_norm**2

        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]

        # Update rotation on the SO(3) manifold
        T_Xb[:-1,:-1] = rot_Xb.dot(exp_omega)
        # Update translation on the manifold
        T_Xb[:-1,-1:] = trans_Xb + A.dot(u)
        trans_Xb, omega_hat, omega = self.Log_map(T_Xb)

        # Update initial state for next iter
        x[0:3,:] = trans_Xb
        x[3:6,:] = omega
        x[6:,:] += delta_x[6:,:]

        return x, T_Xb

    def _get_jacobians_manifold(self, x, T_Xb, phis, N):
        sqrt_sigma = np.linalg.inv(sqrtm(self.meas_noise))
        R_b = T_Xb[:-1,:-1]
        t_b = T_Xb[:-1,-1:]

        # First H_a
        H_a = np.zeros((2*N, 6+2*N))
        for i in range(0,2*N,2):
            H_a[i:i+2,6+i:6+i+2] = sqrt_sigma
            # H_a[i:i+2,6+i:6+i+2] = np.eye(2)

        # Then H_b
        H_b = np.zeros(H_a.shape)
        for j,i in enumerate(range(0,2*N,2)):
            # Get the current estimates of the landmarks
            theta = x[6+i,0]
            rang = x[6+i+1,0]
            phi = phis[j]
            p = self.polar_to_cart(np.array([[theta],[rang],[phi]]))
            q = R_b.transpose().dot(p - t_b)
            qx = q[0,0]
            qy = q[1,0]
            qz = q[2,0]

            zhat_q = np.array([[-qy/np.sqrt(qx**2+qy**2),
                                qx/np.sqrt(qx**2+qy**2), 0.0],
                               [qx/np.sqrt(qx**2+qy**2+qz**2),
                                qy/np.sqrt(qx**2+qy**2+qz**2),
                                qz/np.sqrt(qx**2+qy**2+qz**2)]])
            q_xb = np.array([[0., -qz, qy, -1.0, 0., 0.],
                             [qz, 0., -qx, 0., -1.0, 0.,],
                             [-qy, qx, 0., 0., 0., -1.0]])
            # Whiten and add to Jacobian
            H_b[i:i+2,0:6] = sqrt_sigma.dot(zhat_q.dot(q_xb))
            # H_b[i:i+2,0:6] = zhat_q.dot(q_xb)

            q_p = R_b.transpose()
            p_mi = np.array([[-rang*np.sin(theta)*np.cos(phi),
                              np.cos(theta)*np.cos(phi)],
                             [rang*np.cos(theta)*np.cos(phi),
                              np.sin(theta)*np.cos(phi)],
                             [0., np.sin(phi)]])
            # Whiten and add
            H_b[i:i+2,6+i:6+i+2] = sqrt_sigma.dot(zhat_q.dot(q_p).dot(p_mi))
            # H_b[i:i+2,6+i:6+i+2] = zhat_q.dot(q_p).dot(p_mi)

        return np.vstack((H_a, H_b))

    def _get_error_b_12(self, x, T_Xb, phis, z_a, z_b, N):
        """
        :param x: state
        :type x: np.array (12+2*N,1)

        :return: predicted range-bearing measurement of landmark from Xa to Xb
        :rtype: np.array (2,1) [theta, range]
        """
        sqrt_sigma = np.linalg.inv(sqrtm(self.meas_noise))
        b_a = np.zeros(z_a.shape)
        b_b = np.zeros(z_b.shape)
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]

        for j,i in enumerate(range(0,2*N,2)):
            polar = np.array([[x[12+i,0]],
                              [x[12+i+1,0]],
                              [phis[j]]])
            b_a[i:i+2,:] = sqrt_sigma.dot(z_a[i:i+2,:] - polar[0:2,:])
            q_i = rot_Xb.transpose().dot(self.polar_to_cart(polar) - trans_Xb)
            b_b[i:i+2,:] = sqrt_sigma.dot(z_b[i:i+2,:] - self.cart_to_polar(q_i))

        return np.vstack((b_a, b_b))

    def _get_sqrt_information(self, S, V):
        """
        Computes the sqrt information matrix as in section V-C in
        the paper.

        :param A_d: SV thresholded A matrix
        :type A_d: np.array (6+2N, 6+2N)

        :return: Square root information matrix R
        :rtype: np.array (6,6)
        """
        Gamma = V.transpose().dot(np.diag(S).transpose()).dot(np.diag(S)).dot(V)
        # Shur's complement
        Gamma_11 = Gamma[0:6,0:6]
        Gamma_12 = Gamma[0:6,6:]
        Gamma_21 = Gamma[6:,0:6]
        Gamma_22 = Gamma[6:,6:]
        try:
            Lambda = Gamma_11 - Gamma_12.dot(np.linalg.inv(Gamma_22)).dot(Gamma_21)
        except np.linalg.LinAlgError:
            rospy.logerr("Singular matrix encountered when computing covariance, skipping...")
            self.is_singular = True
            return None
        L, D, P = ldl(Lambda, lower=False)
        sqrt_inf = np.real(sqrtm(D).dot(L[P].transpose()))

        return sqrt_inf

    def _get_equivalent(self, cov):
        # Computes the equivalent 6 by 6 covariance matrix from a 12 by 12
        k = cov[0,0]**2 + cov[1,0]**2
        J_11 = -cov[1,0]/k
        J_14 = cov[0,0]/k
        J_21 = cov[0,0]*cov[2,1]/(np.sqrt(k)*(k+cov[2,1]**2))
        J_24 = cov[1,0]*cov[2,1]/(np.sqrt(k)*(k+cov[2,1]**2))
        J_28 = -np.sqrt(k)/(k+cov[2,1]**2)
        J_38 = cov[2,2]/(cov[2,1]**2+cov[2,2]**2)
        J_39 = -cov[2,1]/(cov[2,1]**2+cov[2,2]**2)
        J = np.array([[J_11,0,0,J_14,0,0,0,0,0],
                      [J_21,0,0,J_24,0,0,0,J_28,0],
                      [0,0,0,0,0,0,0,J_38,J_39]])
        top = np.hstack((np.zeros((3,9)),np.eye(3)))
        bottom = np.hstack((J,np.zeros((3,3))))
        Jacobian = np.vstack((top,bottom))

        return Jacobian.dot(cov).dot(Jacobian.transpose())

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
        # sonar_constraint.pose.covariance = R.ravel().tolist()
        sonar_constraint.pose.covariance = np.eye(6).ravel().tolist()
        self.pose_constraint_pub.publish(sonar_constraint)
        self.tf_pub.sendTransform((trans_Xb[0,0], trans_Xb[1,0], trans_Xb[2,0]),
                                  quat, rospy.Time.now(),
                                  "bundle_adjustment/sonar_pose_constraint",
                                  # "slam/dead_reckoning/sonar_pose")
                                  "rexrov/sonar_pose")

    def _publish_pose(self, R, T_Xb):
        """
        TODO
        """
        if not self.is_pose_init:
            trans, rot = self.tf_listener.lookupTransform('/world',
                                                        '/rexrov/sonar_pose',
                                                        rospy.Time(0))
            pose = tf.transformations.quaternion_matrix(rot)
            pose[:-1, -1] = np.asarray(trans)
            self.sonar_pose  = pose
            self.is_pose_init = True

        pose = self.sonar_pose.dot(T_Xb)
        self.sonar_pose = pose
        quat = tf.transformations.quaternion_from_matrix(pose)
        quat = self.normalize_quat(quat)
        trans_Xb = pose[:-1,-1:]

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
                                  # "/rexrov/sonar_pose")
                                  "/world")
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
        # sonar_pose.covariance = R.ravel().tolist()
        sonar_pose.covariance = np.eye(6).ravel().tolist()
        sonar_odom = Odometry()
        # sonar_odom.header.frame_id = "/rexrov/sonar_pose"
        sonar_odom.header.frame_id = "/world"
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
        pointcloud = [[], [], []]
        # Append points to cloud
        for j,i in enumerate(range(12,state.shape[0],2)):
            cart = self.polar_to_cart(np.array([[state[i,0]],
                                                [state[i+1,0]],
                                                [phis[j]]]))
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
                                                              '/slam/optimized/sonar_pose')
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

    def _update_transform_6(self,Xa,x):
        trans = np.array([[x[0,0]],[x[1,0]],[x[2,0]]])
        # y p r in state
        # quat = tf.transformations.quaternion_from_euler(x[5,0], x[4,0], x[3,0])
        quat = tf.transformations.quaternion_from_euler(x[3,0], x[4,0], x[5,0])
        T_Xb = tf.transformations.quaternion_matrix(quat)
        T_Xb[:-1,-1:] = trans

        return T_Xb

    def _update_transform_12(self, x):
        """
        Update T_Xb
        """
        T_Xb = np.eye(4)
        T_Xb[:-1,-1:] = np.array([[x[9,0]],[x[10,0]],[x[11,0]]])
        T_Xb[:-1,:-1] = np.array([[x[0,0], x[3,0], x[6,0]],
                                  [x[1,0], x[4,0], x[7,0]],
                                  [x[2,0], x[5,0], x[8,0]]])

        return T_Xb

    def _get_jacobians_12(self, x, T_Xb, phis, N):
        sqrt_sigma = np.linalg.inv(sqrtm(self.meas_noise))
        R = T_Xb[:-1,:-1]
        t = T_Xb[:-1,-1:]
        tx = t[0,0]
        ty = t[0,0]
        tz = t[0,0]

        # First H_a
        H_a = np.zeros((2*N, 12+2*N))
        for i in range(0,2*N,2):
            H_a[i:i+2,12+i:12+i+2] = sqrt_sigma

        # Then H_b
        H_b = np.zeros(H_a.shape)
        for j,i in enumerate(range(0,2*N,2)):
            # Get the current estimates of the landmarks
            theta = x[12+i,0]
            rang = x[12+i+1,0]
            phi = phis[j]
            p = self.polar_to_cart(np.array([[theta],[rang],[phi]]))
            px = p[0,0]
            py = p[1,0]
            pz = p[2,0]
            q = R.transpose().dot(p - t)
            qx = q[0,0]
            qy = q[1,0]
            qz = q[2,0]

            zhat_q = np.array([[-qy/np.sqrt(qx**2+qy**2),
                                qx/np.sqrt(qx**2+qy**2), 0.0],
                               [qx/np.sqrt(qx**2+qy**2+qz**2),
                                qy/np.sqrt(qx**2+qy**2+qz**2),
                                qz/np.sqrt(qx**2+qy**2+qz**2)]])
            q_xb = np.array([[px-tx,py-ty,pz-tz,0,0,0,0,0,0,-R[0,0],-R[1,0],-R[2,0]],
                             [0,0,0,px-tx,py-ty,pz-tz,0,0,0,-R[0,1],-R[1,1],-R[2,1]],
                             [0,0,0,0,0,0,px-tx,py-ty,pz-tz,-R[0,2],-R[1,2],-R[2,2]]])
            # Whiten and add to Jacobian
            H_b[i:i+2,0:12] = sqrt_sigma.dot(zhat_q.dot(q_xb))

            q_p = R.transpose()
            p_mi = np.array([[-rang*np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi)],
                             [rang*np.cos(theta)*np.cos(phi),np.sin(theta)*np.cos(phi)],
                             [0., np.sin(phi)]])
            # Whiten and add
            H_b[i:i+2,12+i:12+i+2] = sqrt_sigma.dot(zhat_q.dot(q_p).dot(p_mi))

        return np.vstack((H_a, H_b))

    def _init_state_12(self, landmarks, Xb, T_Xb, N):
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

    def _init_state_6(self, landmarks, Xb, T_Xb, N):
        x = np.zeros((6+2*N, 1))
        z_a = np.zeros((2*N, 1))
        z_b = np.zeros((2*N, 1))
        # X Y Z Y P R
        # quat = tf.transformations.quaternion_from_matrix(T_Xb)
        roll, pitch, yaw = tf.transformations.euler_from_matrix(T_Xb)
        x[0:6,0:1] = np.array([[T_Xb[0,-1]],[T_Xb[1,-1]],[T_Xb[2,-1]],
                               [yaw],[pitch],[roll]])
                               # [roll],[pitch],[yaw]])

        i = 0
        for l in landmarks:
            x[6+i:6+i+2,:] = l.polar_img1
            z_a[i:i+2,:] = l.polar_img1
            z_b[i:i+2,:] = l.polar_img2
            i += 2

        return (x, z_a, z_b)

    def _get_covariance_12(self, S, V):
        """
        Computes the sqrt information matrix as in section V-C in
        the paper.

        :param A_d: SV thresholded A matrix
        :type A_d: np.array (6+2N, 6+2N)

        :return: Square root information matrix R
        :rtype: np.array (6,6)
        """
        Gamma = V.transpose().dot(np.diag(S).transpose()).dot(np.diag(S)).dot(V)
        # Shur's complement
        Gamma_11 = Gamma[0:12,0:12]
        Gamma_12 = Gamma[0:12,12:]
        Gamma_21 = Gamma[12:,0:12]
        Gamma_22 = Gamma[12:,12:]
        try:
            Lambda = Gamma_11 - Gamma_12.dot(np.linalg.inv(Gamma_22)).dot(Gamma_21)
            covariance = np.linalg.inv(Lambda)
        except np.linalg.LinAlgError:
            rospy.logerr("Singular matrix encountered when computing covariance, skipping...")
            self.is_singular = True
            return None
        # Compute equivalent 6 by 6 covariance
        covariance = self._get_equivalent(covariance)

        return covariance

    def _get_error_b(self, x, T_Xb, phis, z_a, z_b, N):
        """
        :param x: state
        :type x: np.array (12+2*N,1)

        :return: predicted range-bearing measurement of landmark from Xa to Xb
        :rtype: np.array (2,1) [theta, range]
        """
        sqrt_sigma = np.linalg.inv(sqrtm(self.meas_noise))
        b_a = np.zeros(z_a.shape)
        b_b = np.zeros(z_b.shape)
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]

        for j,i in enumerate(range(0,2*N,2)):
            polar = np.array([[x[6+i,0]],
                              [x[6+i+1,0]],
                              [phis[j]]])
            b_a[i:i+2,:] = sqrt_sigma.dot(z_a[i:i+2,:] - polar[0:2,:])
            q_i = rot_Xb.transpose().dot(self.polar_to_cart(polar) - trans_Xb)
            b_b[i:i+2,:] = sqrt_sigma.dot(z_b[i:i+2,:] - self.cart_to_polar(q_i))

        return np.vstack((b_a, b_b))
