#!/usr/bin/python2.7
"""
Landmark detection node for the FS sonar SLAM pipeline,
following the methods used in the paper.
"""
import rospy
import tf
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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
    def __init__(self, keypts1=None, keypts2=None, map1=None, map2=None,
                 key1=None, key2=None, T_Xb=None, Xb=None, Xa=None, test=False):

        if not test:
            self.pixels_img1 = tuple(np.flip(np.round(keypts1[key1].pt).astype(int)))
            self.pixels_img2 = tuple(np.flip(np.round(keypts2[key2].pt).astype(int)))
            # ground truth for m_i from img1(X_a) and img2(X_b)
            self.cart_img1 = np.array([[map1[0][self.pixels_img1]],
                                       [map1[1][self.pixels_img1]],
                                       [map1[2][self.pixels_img1]]])
            self.cart_img2 = np.array([[map2[0][self.pixels_img2]],
                                       [map2[1][self.pixels_img2]],
                                       [map2[2][self.pixels_img2]]])
            self.is_shit = False
            if (self.cart_img1 == 0).any() or (self.cart_img2 == 0).any():
                self.is_shit = True
            # init elevation and phi to 0
            self.phi = 0.0
            self.elevation = 0.0
            # measurements in polar coordinates
            self.polar_img1 = self.cart_to_polar(self.cart_img1)
            # TODO(aldoteran): check if simulated measurements can be fixed
            self.polar_img2 = self.cart_to_noisy_polar(self.cart_img1, T_Xb)
            # self.polar_img2 = self.cart_to_polar(self.cart_img2)
            # ground truth for phi, for debugging
            self.real_phi = np.arcsin(self.cart_img1[2,0]/self.polar_img1[1,0])
            # relative pose between img1 and img2 T_Xb
            self.rel_pose = T_Xb
            # pose from img2 (Xa)
            self.Xa = Xa
            # pose from img2 (Xb)
            self.Xb = Xb
        else:
            self.cart_img1 = None
            self.cart_img2 = None
            self.phi = 0.0
            self.elevation = 0.0
            # measurements in polar coordinates
            self.polar_img1 = None
            self.polar_img2 = None
            # ground truth for phi, for debugging
            self.real_phi = None
            # relative pose between img1 and img2 T_Xb
            self.rel_pose = None
            # pose from img2 (Xa)
            self.Xa = None
            # pose from img2 (Xb)
            self.Xb = None

    def update_phi(self, phi):
        # Updates phi and elevation attributes (wrt X_a/img1)
        self.phi = phi
        self.elevation = self.polar_img1[1,0] * np.sin(phi)

    def cart_to_polar(self, cart):
        return np.array([[np.arctan2(cart[1,0],cart[0,0])],
                         [np.sqrt(cart[0,0]**2 + cart[1,0]**2 + cart[2,0]**2)]])

    def cart_to_noisy_polar(self, p_i, T_Xb):
        cart = T_Xb[:-1,:-1].transpose().dot(p_i - T_Xb[:-1,-1:])
        theta = np.arctan2(cart[1,0],cart[0,0]) + np.random.normal(0, 5e-6)
        # theta = np.arctan2(cart[1,0],cart[0,0])
        dist = np.sqrt(cart[0,0]**2+cart[1,0]**2+cart[2,0]**2) + np.random.normal(0, 1.5e-5)
        # # dist = np.sqrt(cart[0,0]**2+cart[1,0]**2+cart[2,0]**2)

        return np.array([[theta],[dist]])


class LandmarkDetector:
    """
    Handles feature dectection and matching for sonar images.

    :param features: type of features to detect
    :type features: str
    """
    BUFF_LEN = 10
    MIN_MATCHES_THRESH = 8

    def __init__(self, verbose=True, debug=False):
        self.detector = cv2.AKAZE_create()
        self.bridge = CvBridge()
        self.is_verbose = verbose
        self.is_debug = debug
        self.is_init = False
        self.img_buff = []
        self.cart_map_buff = []
        self.pcl_buff = []

        #### SUBSCRIBERS ####
        self.tf_listener = tf.TransformListener(cache_time=rospy.Duration(20))
        rospy.Subscriber('/rexrov/depth/image_raw_raw_sonar',
                         Image, self._image_cb)
        rospy.Subscriber('/rexrov/depth/image_raw_depth_raw_sonar',
                         Image, self._range_cb)
        #### PUBLISHERS ####
        self.image_pub = rospy.Publisher('/features_debug', Image,
                                         queue_size=1)
        self.tf_pub = tf.TransformBroadcaster()

    def _image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        img = np.uint8(255 * (img / np.max(img)))
        self.img_buff.append((img,msg.header.stamp,msg.header.seq))
        # if len(self.img_buff) >= self.BUFF_LEN:
            # self.img_buff.pop(-1)

    def _range_cb(self, msg):
        """
        Callback for the ground truth of the sonar measurements. Image
        is a 3D array with channels (range, swath, depth).
        """
        img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.cart_map_buff.append((img[:,:,0],
                                   img[:,:,1],
                                   img[:,:,2],
                                   msg.header.stamp,
                                   msg.header.seq))
        # if len(self.cart_map_buff) >= self.BUFF_LEN:
            # self.cart_map_buff.pop(-1)

    def extract_n_match(self, imgs):
        """
        Extract the features from the input image.

        :params imgs: pair of images to extract and match features from
        :type imgs: list of tuples [(img1,stamp), (img2,stamp)]

        :return: extracted keypoints and correspondances
        :rtype: list [keypoints_img1, keypoints_img2, cv2.BFMatcher.knnMatch]

        :return: None if not enough matches were found
        :rtype: None
        """
        (keypts1, descript1) = self.detector.detectAndCompute(imgs[0][0], None)
        (keypts2, descript2) = self.detector.detectAndCompute(imgs[1][0], None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(descript1, descript2, k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good_matches.append([m])

        # Use the findHomography method to find inliers
        if len(good_matches) > self.MIN_MATCHES_THRESH:
            src_pts = np.float32([keypts1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([keypts2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = np.expand_dims(np.asarray(good_matches)[mask==1], 1).tolist()
        else:
            rospy.logwarn("Not enough matches for model ({}).".format(len(good_matches)))
            return None

        feat_img = cv2.drawMatchesKnn(imgs[0][0], keypts1,
                                      imgs[1][0], keypts2,
                                      inliers, None, flags=2)

        # For debugging
        if self.is_verbose:
            rospy.logwarn("keypoints Img1: {}, descriptors: {}".format(len(keypts1), descript1.shape))
            rospy.logwarn("keypoints Img2: {}, descriptors: {}".format(len(keypts2), descript2.shape))
            # publish extracted features
            img_msg = self.bridge.cv2_to_imgmsg(feat_img, encoding="passthrough")
            self.image_pub.publish(img_msg)

        return [keypts1, keypts2, inliers]

    def _relative_pose(self, imgs):
        """
        Compute the relative pose T_Xb between two frames. Uses the timestamp of each
        image to look for the TF wrt to the world frame.

        :params imgs: pair of images to extract and match features from
        :type imgs: list of tuples [(img1,stamp), (img2,stamp)]

        :return: relative pose T_Xb expressed as an homogenous transformation matrix
        :rtype: np.array (4,4)
        """
        sonar_frame = 'rexrov/sonar_pose'
        # try:
            # # self.tf_listener.waitForTransform('/world', '/slam/optimized/sonar_pose',
                                                # # imgs[0][1], rospy.Duration(0.2))
            # trans_Xa, rot_Xa = self.tf_listener.lookupTransform('/world',
                                                                # '/slam/optimized/sonar_pose',
                                                                # imgs[0][1])
            # trans_Xb, rot_Xb = self.tf_listener.lookupTransform('/world',
                                                                # '/slam/optimized/sonar_pose',
                                                                # imgs[1][1])
        # except:
            # rospy.logwarn("Using IMU odometry instead of optimized pose.")
        # try:
            # # get poses and compute T_xb from the IMU odometry.
            # self.tf_listener.waitForTransform('/world', '/slam/dead_reckoning/sonar_pose',
                                                # imgs[1][1], rospy.Duration(0.2))
            # trans_Xb, rot_Xb = self.tf_listener.lookupTransform('/world',
                                                                # '/slam/dead_reckoning/sonar_pose',
                                                                # imgs[1][1])
            # trans_Xa, rot_Xa = self.tf_listener.lookupTransform('/world',
                                                                # '/slam/dead_reckoning/sonar_pose',
                                                                # imgs[0][1])
        # except:
            # return
        # # self.is_init = True

        # Xa = tf.transformations.quaternion_matrix(rot_Xa)
        # Xa[:-1, -1] = np.asarray(trans_Xa)
        # Xb = tf.transformations.quaternion_matrix(rot_Xb)
        # Xb[:-1, -1] = np.asarray(trans_Xb)
        # T_Xb = np.linalg.inv(Xa).dot(Xb)

        # if self.is_verbose:
        try:
            trans_Xa, rot_Xa = self.tf_listener.lookupTransform('/world',
                                                                sonar_frame,
                                                                imgs[0][1])
            trans_Xb, rot_Xb = self.tf_listener.lookupTransform('/world',
                                                                sonar_frame,
                                                                imgs[1][1])
            Xa_true = tf.transformations.quaternion_matrix(rot_Xa)
            Xa_true[:-1, -1] = np.asarray(trans_Xa)
            Xb_true = tf.transformations.quaternion_matrix(rot_Xb)
            Xb_true[:-1, -1] = np.asarray(trans_Xb)
            T_Xb_true = np.linalg.inv(Xa_true).dot(Xb_true)
            rot_Xb = np.copy(T_Xb_true)
            self.tf_pub.sendTransform((T_Xb_true[0,-1], T_Xb_true[1,-1], T_Xb_true[2,-1]),
                                    tf.transformations.quaternion_from_matrix(rot_Xb),
                                    imgs[1][1],
                                    "/Xb",
                                    "/rexrov/forward_sonar_optical_frame")
        except:
            return

        return (T_Xb_true, Xb_true, Xa_true)
        # return (T_Xb, Xb, Xa)
        # for debugging
        # return (T_Xb, Xb, Xa, T_Xb_true, Xb_true, Xa_true)

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
        try:
            T_Xb, Xb, Xa = self._relative_pose(imgs)
        # for debugging
            # T_Xb, Xb, Xa, T_true, Xb_true, Xa_true = self._relative_pose(imgs)
        except:
            return

        self.landmarks = [Landmark(features[0], features[1], maps[0], maps[1],
                                   match[0].queryIdx, match[0].trainIdx,
                                   T_Xb, Xb, Xa, test=False)
                                   # T_true, Xb, Xa, test=False)
                          for match in features[2]]
        inliers = []
        for i,l in enumerate(self.landmarks):
            if l.is_shit:
                continue
            inliers.append(i)
        # get rid of poorly contrained landmarks
        self.landmarks = np.asarray(self.landmarks)[inliers].tolist()
        dist_list = np.zeros(len(self.landmarks))
        for i in range(len(self.landmarks) - 1):
            dist_list[i] = np.linalg.norm(self.landmarks[i].polar_img1 - self.landmarks[i+1].polar_img1)
        self.landmarks = np.asarray(self.landmarks)[dist_list > 0.30].tolist()

        return self.landmarks

def main():
    import time
    rospy.init_node('landmark_detection')
    detector = LandmarkDetector(verbose=True)
    rospy.loginfo("Initializing landmark detection.")
    rospy.sleep(3.0)
    # initialize with two images first
    if len(detector.img_buff) >= 2:
        img1 = detector.img_buff.pop(0)
        img2 = detector.img_buff.pop(0)
        detector.extract_n_match([img1, img2])
    while not rospy.is_shutdown():
        tic = time.time()
        if len(detector.img_buff) >= 1:
            img1 = img2
            img2 = detector.img_buff.pop(0)
            detector.extract_n_match([img1, img2])
            toc = time.time()
            rospy.loginfo("Computed landmarks in {} seconds".format(toc-tic))

if __name__ == '__main__':
    main()
