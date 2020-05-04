#!/usr/bin/python2.7
"""
Class to handle the Two-View Bundle Adjustment algorithm
explained in the paper "Degeneracy-Aware Imaging Sonar
Simultaneous Localization and Mapping" by Westman and Kaess.
"""
import numpy as np
from scipy.linalg import ldl, sqrtm

import tf
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseWithCovariance, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

class BundleAdjuster:
    """
    TBA
    """
    def __init__(self, verbose=True, test=False):
        self.is_test = test
        self.verbose = verbose
        self.phi_range = np.arange(-0.54977, 0.54977, 0.017453)
        #### PUBLISHERS ####
        self.pose_pub = rospy.Publisher('/bundle_adjustment/sonar_pose',
                                        PoseStamped,
                                        queue_size=1)
        self.pose_constraint_pub = rospy.Publisher('/bundle_adjustment/sonar_constraint',
                                                   PoseStamped,
                                                   queue_size=1)
        self.odom_pub = rospy.Publisher('/sonar_odometry', Odometry,
                                        queue_size=1)
        self.true_odom_pub = rospy.Publisher('/true_sonar_odometry', Odometry,
                                        queue_size=1)
        self.tf_pub = tf.TransformBroadcaster()

    def compute_constraint(self, landmarks, sigma=np.diag((0.1,0.1))):
        """
        Compute sonar constraint using landmarks seen in two
        sonar images.

        :param landmarks: list of N landmarks
        :type landmarks: list [Landmark_1,...,Landmark_N]

        :return:
        """
        N = len(landmarks)
        epsilon = 1e-6
        #TODO: import from config
        sqrt_sigma = np.linalg.inv(sqrtm(np.diag((0.01,0.01))))
        # init with best info up until this point
        T_Xb = landmarks[0].rel_pose
        Xb = landmarks[0].Xb
        Xa = landmarks[0].Xa
        x_init, z_a, z_b = self._init_state(landmarks, Xb, N)
        delta_old = np.zeros(x_init.shape)
        error_old = 9e6
        # Gauss-Newton NLS optimization
        for it in range(10):
            # (1) Direct search for phi_star
            phis = self._opt_phi_search(x_init, z_b, T_Xb, sigma, N)
            # ground truth, for debugging
            # phis = [l.real_phi for l in landmarks]
            # (2) Compute whitened Jacobian A and error vector b
            A = self._get_jacobians(x_init, T_Xb, phis, sqrt_sigma, N)
            b = self._get_error_b(x_init, T_Xb, phis, z_a, z_b, sqrt_sigma, N)
            # (3) SVD of A and thresholding of singular values
            U, S, V = np.linalg.svd(A, full_matrices=False)
            # S[S < 1.0] = 0.0

            # (4) Update initial state
            A_d = U.dot(np.diag(S)).dot(V.transpose())
            delta_new = V.dot(np.linalg.pinv(np.diag(S), True)) \
                         .dot(U.transpose()).dot(b)
            x_new = x_init + delta_new
            error = np.linalg.norm(delta_new - delta_old)
            # (5) Check if converged
            if self.verbose:
                rospy.logwarn("Update error for iter {}: {}".format(it, error))
            if error < epsilon:
                break
            # (6) Update values and iterate
            delta_old = np.copy(delta_new)
            error_old = error
            x_init = np.copy(x_new)
            T_Xb = self._update_transform(Xa, x_init)
            rospy.logwarn("Resulting relative pose: {}".format(T_Xb))

        # Return the sate if in test mode
        if self.is_test:
            return (x_new, T_Xb, phis, S, error)

        sqrt_info = self._get_sqrt_information(A_d)
        self._publish_pose(x_new, sigma)
        self._publish_pose_constraint(T_Xb)
        self._publish_true_odom(Xb)

    def _init_state(self, landmarks, Xb, N):
        """
        Initialize state.

        :param landmarks: list of landmarks
        :type landmarks: list [Landmark_1,...,Landmark_N]

        :param T_Xb: relative pose btwn pose Xa and Xb represented by a
                     4 by 4 rigid transformation matrix
        :type T_Xb: np.array (4,4)

        :return: initial state vector
        :rtype: np.array (6+2*N, 1)
        """
        x = np.zeros((6+2*N, 1))
        z_a = np.zeros((2*N, 1))
        z_b = np.copy(z_a)
        # X Y Z R P Y? or X Y Z Y P R?
        quat = tf.transformations.quaternion_from_matrix(Xb)
        euler = tf.transformations.euler_from_quaternion(quat)
        x[0:6,0:1] = np.array([[Xb[0,-1]],[Xb[1,-1]],[Xb[2,-1]],
                             [euler[2]],[euler[1]],[euler[0]]])
        i = 0
        for l in landmarks:
            x[6+i:6+i+2,:] = l.polar_img1
            z_a[i:i+2,:] = l.polar_img1
            z_b[i:i+2,:] = l.polar_img2
            i += 2

        return (x, z_a, z_b)

    def _opt_phi_search(self, x, z_b, T_Xb, sigma, N):
        """
        Search for optimal phi using the list of phis in phi_range.
        """
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]
        phis = []

        for i in range(0,2*N,2):
            best_phi = 1.0
            old_error = 9999
            z_bi = z_b[i:i+2,:]
            polar = x[6+i:6+i+2,:]
            for phi in self.phi_range:
                q_i = rot_Xb.transpose().dot(self._project_coords(polar, phi) - trans_Xb)
                innov = self.cart_to_polar(q_i) - z_bi
                error = innov.transpose().dot(np.linalg.inv(sigma)).dot(innov)
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
        trans = np.array([[x[0,0]],[x[1,0]],[x[2,0]]])
        # Y P R in state
        quat = tf.transformations.quaternion_from_euler(x[5,0], x[4,0], x[3,0])
        Xb = tf.transformations.quaternion_matrix(quat)
        Xb[:-1,-1:] = trans

        return np.linalg.inv(Xa).dot(Xb)

    def _predict_hb(self, T_Xb, cart):
        return T_Xb[:-1,:-1].transpose().dot(cart - T_Xb[:-1, -1:])

    def _get_jacobians(self, x, T_Xb, phis, sqrt_sigma, N):
        """
        Jacobians from the paper
        """
        H_A = np.hstack((np.zeros((2*N,6)), np.eye(2*N,2*N)))

        H_B = np.zeros((2*N, 6+2*N))
        for j,i in enumerate(range(0,N,2)):
            polar = np.array([[x[6+i,0]],
                              [x[6+i+1,0]],
                              [phis[j]]])
            p = self.polar_to_cart(polar)
            q = self._predict_hb(T_Xb, p)

            zhat_q = np.array([[-q[1,0]/np.sqrt(q[0,0]**2+q[1,0]**2),
                                q[0,0]/np.sqrt(q[0,0]**2+q[1,0]**2), 0.0],
                            [q[0,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2),
                                q[1,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2),
                                q[2,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2)]])
            q_xb = np.array([[0.0, -q[2,0], q[1,0]],
                             [q[2,0], 0.0, -q[0,0]],
                             [-q[1,0], q[0,0], 0.0]])
            H_B[i:i+2,0:6] = zhat_q.dot(np.hstack((q_xb, -np.eye(3))))

            q_p = T_Xb[:-1, :-1].transpose()
            p_mi = np.array([[-polar[1,0]*np.sin(polar[0,0])*np.cos(polar[2,0]),
                              np.cos(polar[0,0])*np.cos(polar[2,0])],
                            [polar[1,0]*np.cos(polar[0,0])*np.cos(polar[2,0]),
                             np.sin(polar[0,0])*np.cos(polar[2,0])],
                            [0.0, np.sin(polar[2,0])]])
            H_B[i:i+2,6+2*i:8+2*i] = zhat_q.dot(q_p).dot(p_mi)
            H_B

        return np.vstack((H_A, H_B))

    def _get_error_b(self, x, T_Xb, phis, z_a, z_b, sqrt_sigma, N):
        """
        TODO

        :param x: state
        :type x: np.array (12+2*N,1)

        :return: predicted range-bearing measurement of landmark from Xa to Xb
        :rtype: np.array (2,1) [theta, range]
        """
        b_a = np.zeros((x.shape[0]-6, 1))
        b_b = np.copy(b_a)
        rot_Xb = T_Xb[:-1, :-1]
        trans_Xb = T_Xb[:-1, -1:]

        for j,i in enumerate(range(0,2*N,2)):
            polar = np.array([[x[6+i,0]],
                              [x[6+i+1,0]],
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
        Gamma_11 = Gamma[0:6,0:6]
        Gamma_12 = Gamma[0:6,6:]
        Gamma_21 = Gamma[6:,0:6]
        Gamma_22 = Gamma[6:,6:]
        Lambda = Gamma_11 - Gamma_12.dot(np.linalg.inv(Gamma_22)).dot(Gamma_21)
        L, D, P = ldl(Lambda, lower=True)
        sqrt_inf = sqrtm(D).transpose().dot(L[P,:].transpose())

        return sqrt_inf

    def _publish_pose_constraint(self, T_Xb):
        """
        Publish the sonar pose constraint to add in the factor graph.
        Contains the relative transformation between two sonar images.

        :param T_Xb: Relative transformation matrix
        :type T_Xb: np.array (4,4)
        """
        trans_Xb = T_Xb[:-1,-1:]
        quat = tf.transformations.quaternion_from_matrix(T_Xb)
        quat = self.normalize_quat(quat)
        # Publish pose
        sonar_constraint = PoseStamped()
        sonar_constraint.header.frame_id = "/sonar_pose_constraint"
        sonar_constraint.header.stamp = rospy.Time.now()
        sonar_constraint.pose.position.x = trans_Xb[0,0]
        sonar_constraint.pose.position.y = trans_Xb[1,0]
        sonar_constraint.pose.position.z = trans_Xb[2,0]
        sonar_constraint.pose.orientation.x = quat[0]
        sonar_constraint.pose.orientation.y = quat[1]
        sonar_constraint.pose.orientation.z = quat[2]
        sonar_constraint.pose.orientation.w = quat[3]
        self.pose_constraint_pub.publish(sonar_constraint)

    def _publish_pose(self, x, sigma):
        """
        TODO
        """
        trans_Xb = np.array([[x[0,0]],
                             [x[1,0]],
                             [x[2,0]]])
        quat = tf.transformations.quaternion_from_euler(x[5,0],
                                                        x[4,0],
                                                        x[3,0])
        quat = self.normalize_quat(quat)
        # Publish pose
        sonar_pose = PoseStamped()
        sonar_pose.header.frame_id = "/sonar_estimate"
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
                                  "/sonar_estimate",
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
        sonar_odom = Odometry()
        sonar_odom.header.frame_id = "/world"
        sonar_odom.child_frame_id = "/sonar_estimate"
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
        sonar_odom.child_frame_id = "/sonar_estimate"
        sonar_odom.header.stamp = rospy.Time.now()
        sonar_odom.pose = sonar_pose
        self.true_odom_pub.publish(sonar_odom)

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

    def _create_marker(self, landmark, idx, imgs):
        # create markers to display in rviz for debugging
        marker = Marker()
        marker.header.stamp = imgs[0][1]
        # marker.header.frame_id = '/Xb'
        marker.header.frame_id = '/rexrov/forward_sonar_optical_frame'
        marker.ns = 'landmark'
        marker.id = idx
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.g = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.5)
        marker.pose.position.x = landmark.cart_img1[1,0]
        marker.pose.position.y = landmark.cart_img1[2,0]
        marker.pose.position.z = landmark.cart_img1[0,0]

        return marker

    def _create_marker_est(self, landmark, idx, imgs):
        # create markers to display in rviz for debugging
        marker = Marker()
        marker.header.stamp = imgs[0][1]
        marker.header.frame_id = '/Xb'
        marker.ns = 'landmark'
        marker.id = idx
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.5)
        marker.pose.position.x = landmark.cart_img2[1,0]
        marker.pose.position.y = landmark.elevation
        # marker.pose.position.y = landmark.cart_img2[2,0]
        marker.pose.position.z = landmark.cart_img2[0,0]

        return marker















