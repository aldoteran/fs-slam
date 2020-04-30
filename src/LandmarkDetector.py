#!/usr/bin/python2.7
"""
Landmark detection node for the FS sonar SLAM pipeline,
following the methods used in the paper.
"""
import rospy
import tf
import cv2
import timeit
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image#, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from BundleAdjuster import BundleAdjuster

__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

class Landmark:
    """
    Makes life much easier. Handles all the information
    for a Landmark between a pair of images.

    :param keypts1: Extracted keypoints from older image
    :type keypts1: list of cv2.KeyPoint

    :param keypts2: Extracted keypoints from latest image
    :type keypts2: list of cv2.KeyPoint

    :param key1: Keypoint index corresponding to the matched
                 feature in the older image
    :type key1: int

    :param key2: Keypoint index corresponding to the matched
                 feature in the latest image
    :type key2: int

    :param map1: Cartesian mappings for img1
    :type map1: tuple (x,y,z,stamp)

    :param map2: Cartesian mappings for img2
    :type map2: tuple (x,y,z,stamp)

    """
    def __init__(self, keypts1, keypts2, map1, map2, key1, key2, T_Xb, Xb, Xa):
        self.pixels_img1 = tuple(np.flip(np.round(keypts1[key1].pt).astype(int)))
        self.pixels_img2 = tuple(np.flip(np.round(keypts2[key2].pt).astype(int)))
        # ground truth for m_i from img1(X_a) and img2(X_b)
        self.cart_img1 = np.array([[map1[0][self.pixels_img1]],
                                   [map1[1][self.pixels_img1]],
                                   [map1[2][self.pixels_img1]]])
        self.cart_img2 = np.array([[map2[0][self.pixels_img2]],
                                   [map2[1][self.pixels_img2]],
                                   [map1[2][self.pixels_img2]]])
        # TODO(aldoteran): maybe this can be done better
        self.is_shit = False
        if (self.cart_img1 == 0).any() or (self.cart_img2 == 0).any():
            self.is_shit = True
        # init elevation and phi to 0
        self.phi = 0.0
        self.elevation = 0.0
        # measurements in polar coordinates
        self.polar_img1 = self.cart_to_polar(self.cart_img1)
        self.polar_img2 = self.cart_to_polar(self.cart_img2)
        # ground truth for phi, for debugging
        self.real_phi = np.arcsin(self.cart_img1[2,0]/self.polar_img1[1,0])
        # relative pose between img1 and img2 T_Xb
        self.rel_pose = T_Xb
        # pose from img2 (Xa)
        self.Xa = Xa
        # pose from img2 (Xb)
        self.Xb = Xb

    def update_phi(self, phi):
        # Updates phi and elevation attributes (wrt X_a/img1)
        self.phi = phi
        self.elevation = self.polar_img1[1,0] * np.sin(phi)

    def project_coords(self, phi):
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
        r = self.polar_img1[1,0]
        theta = self.polar_img1[0,0]

        return np.array([[r * np.cos(theta) * np.cos(phi)],
                         [r * np.sin(theta) * np.cos(phi)],
                         [r * np.sin(phi)]])

    def cart_to_polar(self, cart):
        return np.array([[np.arctan2(cart[1,0],cart[0,0])],
                         [np.sqrt(cart[0,0]**2 + cart[1,0]**2 + cart[2,0]**2)]])


class LandmarkDetector:
    """
    Handles feature dectection and matching for sonar images.

    :param features: type of features to detect
    :type features: str
    """
    # TODO: give options for detection of diff features?
    feat_dict = {'AKAZE': cv2.AKAZE_create(),
                 'SIFT': cv2.xfeatures2d.SIFT_create(),
                 'SURF': cv2.xfeatures2d.SURF_create()}
                 # 'ORB': cv2.ORB()}
    BUFF_LEN = 4

    def __init__(self, features='AKAZE', verbose=True):
        self.detector = self.feat_dict[features]
        self.bridge = CvBridge()
        self.is_verbose = verbose
        self.is_init = False
        self.img_buff = []
        self.cart_map_buff = []
        self.pcl_buff = []

        #### SUBSCRIPTIONS ####
        #TODO(aldoteran): Maybe using a sonar img handler is more modular
        # self.sonar_handler = SimSonarHandler()
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber('/rexrov/depth/image_raw_raw_sonar',
                         Image, self._image_cb)
        rospy.Subscriber('/rexrov/depth/image_raw_depth_raw_sonar',
                         Image, self._range_cb)

        #### PUBLISHERS ####
        self.image_pub = rospy.Publisher('/features_debug', Image,
                                         queue_size=1)
        self.marker_pub = rospy.Publisher('/landmarks', MarkerArray,
                                          queue_size=1)
        self.tf_pub = tf.TransformBroadcaster()

    def _image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        img = np.uint8(255 * (img / np.max(img)))
        stamp = msg.header.stamp
        self.sonar_frame = msg.header.frame_id
        self.img_buff.append((img,stamp))
        if len(self.img_buff) >= self.BUFF_LEN:
            self.img_buff.pop(0)

    def _range_cb(self, msg):
        """
        Callback for the ground truth of the sonar measurements. Image
        is a 3D array with channels (range, swath, depth).
        """
        img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        stamp = msg.header.stamp
        self.cart_map_buff.append((img[:,:,0],
                                   img[:,:,1],
                                   img[:,:,2],
                                   stamp))
        if len(self.cart_map_buff) >= self.BUFF_LEN:
            self.cart_map_buff.pop(0)

    def extact_n_match(self, imgs):
        """
        Extract the features from the input image.

        :params imgs: pair of images to extract and match features from
        :type imgs: list of tuples [(img1,stamp), (img2,stamp)]

        :return: extracted keypoints and correspondances
        :rtype: list [keypoints_img1, keypoints_img2, cv2.BFMatcher.knnMatch]
        """
        tic = timeit.timeit()
        (keypts1, descript1) = self.detector.detectAndCompute(imgs[0][0], None)
        (keypts2, descript2) = self.detector.detectAndCompute(imgs[1][0], None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(descript1, descript2, k=2)

        #TODO(aldoteran): outlier detection, JCBB?
        good_matches = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good_matches.append([m])

        feat_img = cv2.drawMatchesKnn(imgs[0][0], keypts1,
                                      imgs[1][0], keypts2,
                                      good_matches, None, flags=2)

        # For debugging
        if self.is_verbose:
            rospy.logwarn("keypoints Img1: {}, descriptors: {}".format(len(keypts1), descript1.shape))
            rospy.logwarn("keypoints Img2: {}, descriptors: {}".format(len(keypts2), descript2.shape))
            rospy.logwarn("Feature matching computed in: {} seconds".format(timeit.timeit()- tic))
            # publish extracted features
            img_msg = self.bridge.cv2_to_imgmsg(feat_img, encoding="passthrough")
            self.image_pub.publish(img_msg)

        return [keypts1, keypts2, good_matches]

    def generate_landmarks(self, imgs, features, maps):
        """
        Generates a list of Landmark instances and updates each
        one with its optima elevation angle following Eq.(17) in the paper.

        :params imgs: pair of images to extract and match features from
        :type imgs: list of tuples [(img1,stamp), (img2,stamp)]

        :params features: list of keypoints and matches for the features
                          in both images
        :type features: list [keypts_img1, keypts_img2, [cv2.DMatch,...,N]]

        :return: list of landmarks with updated elevation angle
        :rtype: list [Landmark_0,...,Landmark_N]
        """
        T_Xb, Xb, Xa = self._relative_pose(imgs)

        self.landmarks = [Landmark(features[0], features[1], maps[0], maps[1],
                                   match[0].queryIdx, match[0].trainIdx,
                                   T_Xb, Xb, Xa)
                          for match in features[2]]

        #TODO(aldoteran): import from config file
        # VFOV from the depth camera in gazebo is 63 deg, res=0.5deg
        phi_range = np.arange(-0.54977, 0.54977, 0.017453)
        sigma = np.float32(np.diag((0.01, 0.01)))

        inliers = []
        # TODO(aldoteran): fix this shit
        for i,l in enumerate(self.landmarks):
            if l.is_shit:
                continue
            # phi_star = self._opt_phi_search(l, T_Xb, sigma, phi_range, imgs)
            # l.update_phi(phi_star)
            inliers.append(i)
        # get rid of poorly contrained landmarks
        self.landmarks = np.asarray(self.landmarks)[inliers].tolist()

        # for debugging, publish landmark markers
        # if self.is_verbose:
            # landmarkers = MarkerArray()
            # for i,l in enumerate(self.landmarks):
                # # Green ground truth markers
                # landmarkers.markers.append(self._create_marker(l,i, imgs))
                # # Red estimated markers
                # landmarkers.markers.append(self._create_marker_est(l,i+len(self.landmarks), imgs))
            # self.marker_pub.publish(landmarkers)

        return self.landmarks

    def _relative_pose(self, imgs):
        """
        Compute the relative pose T_Xb between two frames. Uses the timestamp of each
        image to look for the TF wrt to the world frame.

        :params imgs: pair of images to extract and match features from
        :type imgs: list of tuples [(img1,stamp), (img2,stamp)]

        :return: relative pose T_Xb expressed as an homogenous transformation matrix
        :rtype: np.array (4,4)
        """
        try:
            # get poses and compute T_xb from the optimized path.
            trans_Xa, rot_Xa = self.tf_listener.lookupTransform('/world',
                                                                '/slam/sonar_optimized_link',
                                                                imgs[0][-1])
            trans_Xb, rot_Xb = self.tf_listener.lookupTransform('/world',
                                                                '/slam/sonar_optimized_link',
                                                                imgs[1][-1])
        except:
            rospy.logwarn("Using IMU odometry instead of optimized pose.")
            # get poses and compute T_xb from the IMU odometry.
            trans_Xa, rot_Xa = self.tf_listener.lookupTransform('/world',
                                                                '/sonar/odom_link',
                                                                imgs[0][-1])
            trans_Xb, rot_Xb = self.tf_listener.lookupTransform('/world',
                                                                '/sonar/odom_link',
                                                                imgs[1][-1])
            self.is_init = True

        Xa = tf.transformations.quaternion_matrix(rot_Xa)
        Xa[:-1, -1] = np.asarray(trans_Xa)
        Xb = tf.transformations.quaternion_matrix(rot_Xb)
        Xb[:-1, -1] = np.asarray(trans_Xb)
        T_Xb = np.linalg.inv(Xa).dot(Xb)

        # for debugging
        if self.is_verbose:
            rot_Xb = np.copy(T_Xb)
            rot_Xb[0:-1,-1] = 0.0
            self.tf_pub.sendTransform((T_Xb[0,-1], T_Xb[1,-1], T_Xb[2,-1]),
                                    tf.transformations.quaternion_from_matrix(rot_Xb),
                                    imgs[1][-1],
                                    "/Xb",
                                    "/rexrov/forward_sonar_optical_frame")

        return (T_Xb, Xb, Xa)

    def _opt_phi_search(self, landmark, T_Xb, sigma, phi_range, imgs):
        """
        Search for optimal phi using the list of phis in phi_range.
        """
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = T_Xb[:-1,-1:]

        best_phi = 1.0
        old_error = 9999
        z_b = landmark.polar_img2
        for phi in phi_range:
            q_i = rot_Xb.transpose().dot(landmark.project_coords(phi) - trans_Xb)
            innov = landmark.cart_to_polar(q_i) - z_b
            error = innov.transpose().dot(np.linalg.inv(sigma)).dot(innov)
            if error < old_error:
                best_phi = phi
                old_error = error

        return best_phi
        # return landmark.real_phi

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

    def _create_marker_proj(self, idx, cart, imgs):
        # create markers to display in rviz for debugging
        marker = Marker()
        marker.header.stamp = imgs[0][1]
        # marker.header.frame_id = '/Xb'
        marker.header.frame_id = '/Xb'
        marker.ns = 'landmark'
        marker.id = idx
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.5)
        marker.pose.position.x = cart[1,0]
        marker.pose.position.y = cart[2,0]
        marker.pose.position.z = cart[0,0]

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

def main():
    rospy.init_node('feature_extraction')
    detector = LandmarkDetector(features='AKAZE')
    bundler = BundleAdjuster()
    rospy.sleep(2)
    # initialize with two first images
    if len(detector.img_buff) >= 2:
        img1 = detector.img_buff.pop(0)
        img2 = detector.img_buff.pop(0)
        maps_img1 = detector.cart_map_buff.pop(0)
        maps_img2 = detector.cart_map_buff.pop(0)
        features = detector.extact_n_match([img1, img2])
        landmarks = detector.generate_landmarks([img1, img2], features,
                                                [maps_img1, maps_img2])
        bundler.compute_constraint(landmarks)
    while not rospy.is_shutdown():
        if len(detector.img_buff) >= 1 & len(detector.cart_map_buff) >=1:
            # drop only first image
            img1 = img2
            img2 = detector.img_buff.pop(0)
            #TODO(aldoteran): assert stamps correspond btwn img and map
            maps_img1 = maps_img2
            maps_img2 = detector.cart_map_buff.pop(0)
            features = detector.extact_n_match([img1, img2])
            landmarks = detector.generate_landmarks([img1, img2], features,
                                                    [maps_img1, maps_img2])
            bundler.compute_constraint(landmarks)

if __name__ == '__main__':
    main()

