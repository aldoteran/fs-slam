#!/usr/bin/python2.7
"""
Class to handle the Two-View Bundle Adjustment algorithm
explained in the paper "Degeneracy-Aware Imaging Sonar
Simultaneous Localization and Mapping" by Westman and Kaess.
"""
import rospy
import tf
import timeit
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped

__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

class BundleAdjuster:
    """
    TBA
    """
    def __init__(self, verbose=True):

        #### PUBLISHERS ####
        self.pose_pub = rospy.Publisher('/sonar_constraint',
                                        PoseWithCovarianceStamped,
                                        queue_size=1)
        self.tf_pub = tf.TransformBroadcaster()

    def compute_constraint(self, landmarks, sigma=np.diag((0.01,0.01))):
        """
        Compute sonar constraint using landmarks seen in two
        sonar images.

        :param landmarks: list of N landmarks
        :type landmarks: list [Landmark_1,...,Landmark_N]

        :return:
        """
        tic = timeit.timeit()

        N = len(landmarks)
        epsilon = 1e-6
        #TODO: better covariance?
        sqrt_sigma = np.linalg.inv(np.sqrt(np.diag((0.0001,0.0001))))
        # init with best info up until this point
        T_Xb = landmarks[0].rel_pose
        x_init = self._init_state(landmarks, T_Xb, N)
        delta_old = np.zeros(x_init.shape)

        #TODO: get all this shit in the same iteration
        z_a = np.asarray([l.polar_img1 for l in landmarks])
        z_a = z_a.reshape((z_a.shape[0]*z_a.shape[1],1))
        z_b = np.asarray([l.polar_img2 for l in landmarks])
        z_b = z_b.reshape((z_b.shape[0]*z_b.shape[1],1))
        # phis = [l.phi for l in landmarks]
        # ground truth, for debugging
        phis = [l.real_phi for l in landmarks]

        for it in range(10):
            # whitened Jacobian A
            A = self._get_jacobians(x_init, phis, sqrt_sigma, N)
            # error vector b
            b = self.whitened_b(x_init, phis, z_a, z_b, sqrt_sigma, N)
            # SVD
            U, S, V = np.linalg.svd(A, full_matrices=False)
            S[S < 2.0] = 0.0
            # state update
            delta_new = V.dot(np.linalg.pinv(np.diag(S))).dot(U.transpose()).dot(b)
            x_new = x_init + delta_new
            error = np.sum(np.linalg.norm(delta_new - delta_old))
            rospy.logwarn("Update error for iter {}: {}".format(it, error))
            if error < epsilon:
                break
            delta_old = np.copy(delta_new)
            x_init = np.copy(x_new)

        toc = timeit.timeit()
        rospy.logwarn("Gauss-Newton executed in {} seconds".format(toc - tic))
        self._publish_pose(x_new, sigma)

    def _get_jacobians(self, x, phis, sqrt_sigma, N):
        """
        TODO
        """
        H_A = np.hstack((np.zeros((2*N,12)), np.eye(2*N,2*N)))

        H_B = np.zeros((2*N, 12+2*N))
        for i in range(0,N,2):
            polar = np.array([[x[12+i,0]],
                              [x[12+i+1,0]],
                              [phis[i/2]]])
            p = self.polar_to_cart(polar)
            q = self._predict_hb(x, p)

            zhat_q = np.array([[-q[1,0]/np.sqrt(q[0,0]**2+q[1,0]**2),
                                q[0,0]/np.sqrt(q[0,0]**2+q[1,0]**2), 0.0],
                            [q[0,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2),
                                q[1,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2),
                                q[2,0]/np.sqrt(q[0,0]**2+q[1,0]**2+q[2,0]**2)]])
            q_xb = np.array([[p[0,0]-x[9,0], p[1,0]-x[10,0], p[2,0]-x[11,0],
                            0.,0.,0.,0.,0.,0.,-x[0,0],-x[1,0],-x[2,0]],
                            [0.,0.,0., p[0,0]-x[9,0], p[1,0]-x[10,0], p[2,0]-x[11,0],
                            0.,0.,0., -x[3,0], -x[4,0], -x[5,0]],
                            [0.,0.,0.,0.,0.,0., p[0,0]-x[9,0], p[1,0]-x[10,0],
                            p[2,0]-x[11,0], -x[6,0], -x[7,0], -x[8,0]]])
            H_B[i:i+2,0:12] = zhat_q.dot(q_xb)

            q_p = np.array([[x[0,0], x[3,0], x[6,0]],
                            [x[1,0], x[4,0], x[7,0]],
                            [x[2,0], x[5,0], x[8,0]]]).transpose()
            p_mi = np.array([[-polar[1,0]*np.sin(polar[0,0])*np.cos(polar[2,0]),
                              np.cos(polar[0,0])*np.cos(polar[2,0])],
                            [polar[1,0]*np.cos(polar[0,0])*np.cos(polar[2,0]),
                             np.sin(polar[0,0])*np.cos(polar[2,0])],
                            [0.0, np.sin(polar[2,0])]])
            H_B[i:i+2,12+2*i:14+2*i] = zhat_q.dot(q_p).dot(p_mi)

        return np.vstack((H_A, H_B))

    def _predict_hb(self, x, cart):
        """
        TODO
        """
        rot_Xb = np.array([[x[0,0], x[3,0], x[6,0]],
                           [x[1,0], x[4,0], x[7,0]],
                           [x[2,0], x[5,0], x[8,0]]])
        trans_Xb = np.array([[x[9,0]],
                             [x[10,0]],
                             [x[11,0]]])

        # return self.from_ROS(rot_Xb.transpose().dot(self.to_ROS(cart) - trans_Xb))
        return rot_Xb.transpose().dot(cart - trans_Xb)

    def whitened_b(self, x, phis, z_a, z_b, sqrt_sigma, N):
        """
        TODO

        :param x: state
        :type x: np.array (12+2*N,1)

        :return: predicted range-bearing measurement of landmark from Xa to Xb
        :rtype: np.array (2,1) [theta, range]
        """
        b_a = np.zeros((x.shape[0]-12, 1))
        b_b = np.copy(b_a)
        rot_Xb = np.array([[x[0,0], x[3,0], x[6,0]],
                           [x[1,0], x[4,0], x[7,0]],
                           [x[2,0], x[5,0], x[8,0]]])
        trans_Xb = np.array([[x[9,0]],
                             [x[10,0]],
                             [x[11,0]]])

        for i in range(0,N,2):
            # polar = self.wrap_to_pi(np.array([[x[12+i,0]],
                                              # [x[12+i+1,0]],
                                              # [phis[i/2]]]))
            polar = np.array([[x[12+i,0]],
                              [x[12+i+1,0]],
                              [phis[i/2]]])
            b_a[i:i+2,:] = z_a[i:i+2,:] - polar[0:2,:]
            q_i = rot_Xb.transpose().dot(self.to_ROS(self.polar_to_cart(polar)) - trans_Xb)
            b_b[i:i+2,:] = self.wrap_to_pi(sqrt_sigma.dot(z_b[i:i+2,:] \
                                           - self.cart_to_polar(self.from_ROS(q_i))))

        return np.vstack((b_a, b_b))

    def to_ROS(self, cart):
        """
        Permutes a numpy array of cartesian coordinates to ROS'
        opitcal frame convention.

        :param cart: cartesian coordinates
        :type cart: np.array (3,1) [Z,X,Y]

        :return: permuted cartesian coordinates
        :rtype: np.array (3,1) [X,Y,Z]
        """
        return np.array([[cart[1,0]],
                         [cart[2,0]],
                         [cart[0,0]]])

    def from_ROS(self, inv_cart):
        """
        Performs the inverse permutation of the to_ROS method.

        :param inv_cart: cartesian coords
        :type inv_cart: np.array (3,1) [X,Y,Z]

        :return: permuted cartesian coordinates
        :rtype: np.array (3,1) [Z,X,Y]
        """
        return np.array([[inv_cart[2,0]],
                         [inv_cart[0,0]],
                         [inv_cart[1,0]]])

    def wrap_to_pi(self, polar):
        return np.array([[(polar[0,0] + np.pi) % (2 * np.pi) - np.pi],
                         [polar[1,0]]])

    def polar_to_cart(self, polar):
        return np.array([[polar[1,0] * np.cos(polar[0,0]) * np.cos(polar[2,0])],
                         [polar[1,0] * np.sin(polar[0,0]) * np.cos(polar[2,0])],
                         [polar[1,0] * np.sin(polar[2,0])]])

    def cart_to_polar(self, cart):
        return np.array([[np.arctan2(cart[1,0],cart[0,0])],
                         [np.sqrt(cart[0,0]**2 + cart[1,0]**2 + cart[2,0]**2)]])

    def _init_state(self, landmarks, T_Xb, N):
        """
        Initialize state.

        :param landmarks: list of landmarks
        :type landmarks: list [Landmark_1,...,Landmark_N]

        :param T_Xb: relative pose btwn pose Xa and Xb represented by a
                     4 by 4 rigid transformation matrix
        :type T_Xb: np.array (4,4)

        :return: initial state vector
        :rtype: np.array (12+2*N, 1)
        """
        x = np.zeros((12+2*N, 1))
        x[0:12,0] = T_Xb[0:-1,:].flatten(order='F')
        land_est = np.asarray([l.polar_img1 for l in landmarks])
        x[12:,:] = np.reshape(land_est, (land_est.shape[0]*land_est.shape[1],1))

        return x

    def _jacobian_Ha(self, landmark, idx, N):
        """
        [DEPRECATED]

        :param landmark: landmark m_i
        :type landmark: Landmark

        :param idx: index i of the landmark
        :type idx: int

        :param N: total number of landmarks in state
        :type N: int

        :return: Jacobian matrix H_Ai
        :rtype: np.array (2*N, 12+2*N)
        """
        idx += 12 # pose offset
        H_Ai = np.zeros((2, 12+2*N))
        H_Ai[:,idx:idx+2] = np.eye(2)

        return H_Ai

    def _jacobian_Hb(self, landmark, idx, N):
        """
        [DEPRECATED]

        :param landmark: landmark m_i
        :type landmark: Landmark

        :param idx: index i of the landmark
        :type idx: int

        :param N: total number of landmarks in state
        :type N: int

        :return: Jacobian matrix H_Ai
        :rtype: np.array (2*N, 12+2*N)
        """
        idx += 12 # pose offset
        # homogeneous transformation
        T_Xb = landmark.rel_pose
        # wrt Xb
        z_hat = landmark.prediction_hb()[0]
        q_x = z_hat[0,0]
        q_y = z_hat[1,0]
        q_z = z_hat[2,0]
        # wrt Xa
        theta = landmark.polar_img1[0,0]
        r = landmark.polar_img1[1,0]
        phi = landmark.phi
        p_x = landmark.cart_img1[0,0]
        p_y = landmark.cart_img1[1,0]
        p_z = landmark.cart_img1[2,0]

        zhat_q = np.array([[-q_y/np.sqrt(q_x**2+q_y**2),
                            q_x/np.sqrt(q_x**2+q_y**2), 0.0],
                           [q_x/np.sqrt(q_x**2+q_y**2+q_z**2),
                            q_y/np.sqrt(q_x**2+q_y**2+q_z**2),
                            q_z/np.sqrt(q_x**2+q_y**2+q_z**2)]])
        q_xb = np.array([[p_x-T_Xb[0,-1], p_y-T_Xb[1,-1], p_z-T_Xb[2,-1],
                          0.,0.,0.,0.,0.,0.,-T_Xb[0,0],-T_Xb[1,0],-T_Xb[2,0]],
                         [0.,0.,0., p_x-T_Xb[0,-1], p_y-T_Xb[1,-1], p_z-T_Xb[2,-1],
                          0.,0.,0., -T_Xb[0,1], -T_Xb[1,1], -T_Xb[2,1]],
                         [0.,0.,0.,0.,0.,0., p_x-T_Xb[0,-1], p_y-T_Xb[1,-1],
                          p_z-T_Xb[2,-1], -T_Xb[0,2], -T_Xb[1,2], -T_Xb[2,2]]])
        q_p = T_Xb[:-1,:-1].transpose()
        p_mi = np.array([[-r*np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi)],
                         [r*np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi)],
                         [0.0, np.sin(phi)]])

        H_Bi = np.zeros((2, 12+2*N))
        H_Bi[:,0:12] = zhat_q.dot(q_xb)
        H_Bi[:,idx:idx+2] = zhat_q.dot(q_p).dot(p_mi)

        return H_Bi

    def _publish_pose(self, x, sigma):
        """
        Publish the sonar pose constraint using output from the optimization

        :param x: optimal state resulting from the GN optimization
        :type x: np.array (12+2*N,1)
        """
        rot_Xb = np.array([[x[0,0], x[3,0], x[6,0], 0.],
                           [x[1,0], x[4,0], x[7,0], 0.],
                           [x[2,0], x[5,0], x[8,0], 0.],
                           [0., 0., 0., 1.]])
        trans_Xb = np.array([[x[9,0]],
                             [x[10,0]],
                             [x[11,0]]])
        quat = self._normalize_quat(tf.transformations.quaternion_from_matrix(rot_Xb))

        #TODO(aldoteran): might have to change axis bc of camera frame convention
        sonar_pose = PoseWithCovarianceStamped()
        sonar_pose.header.frame_id = "/rexrov/forward_sonar_optical_frame"
        sonar_pose.header.stamp = rospy.Time.now()
        sonar_pose.pose.pose.position.x = trans_Xb[0,0]
        sonar_pose.pose.pose.position.y = trans_Xb[1,0]
        sonar_pose.pose.pose.position.z = trans_Xb[2,0]
        sonar_pose.pose.pose.orientation.x = quat[0]
        sonar_pose.pose.pose.orientation.y = quat[1]
        sonar_pose.pose.pose.orientation.z = quat[2]
        sonar_pose.pose.pose.orientation.w = quat[3]

        # publish tf for debugging
        self.tf_pub.sendTransform((trans_Xb[0,0], trans_Xb[1,0], trans_Xb[2,0]),
        # self.tf_pub.sendTransform((trans_Xb[0,0], 0, trans_Xb[2,0]),
                                  quat, rospy.Time.now(),
                                  # self._normalize_quat([0,quat[1],0,quat[3]]),
                                  # rospy.Time.now(),
                                  "/sonar_estimate",
                                  "/rexrov/forward_sonar_optical_frame")

        self.pose_pub.publish(sonar_pose)

    def _normalize_quat(self,quat):
        quat = np.asarray(quat)
        length = np.linalg.norm(quat)

        return (quat/length).tolist()

















