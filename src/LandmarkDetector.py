#!/usr/bin/python2.7
"""
Landmark detection node for the FS sonar SLAM pipeline,
following the methods used in the paper.
"""
import rospy
import tf
import cv2
import random
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

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
    def __init__(self, keypts1, keypts2, map1, map2, key1, key2, T_Xb):

        self.pixels_img1 = tuple(np.flip(np.round(keypts1[key1].pt).astype(int)))
        self.pixels_img2 = tuple(np.flip(np.round(keypts2[key2].pt).astype(int)))
        # ground truth for m_i from img1(X_a) and img2(X_b)
        self.polar_img1 = np.array([[map1[0][self.pixels_img1]],
                                    [map1[1][self.pixels_img1]]])
        self.polar_img2 = np.array([[map2[0][self.pixels_img2]],
                                    [map2[1][self.pixels_img2]]])
        # init elevation and phi to 0
        self.phi = 0.0
        self.elevation = 0.0
        # relative pose between img1 and img2 T_Xb
        self.rel_pose = T_Xb

    def update_phi(self, phi):
        # Updates phi and elevation attributes (wrt X_a/img1)
        self.phi = phi
        self.elevation = self.polar_img1[1,0] * np.sin(phi)

class LandmarkDetector:
    """
    Handles feature dectection and matching for sonar images.

    :param features: type of features to detect
    :type features: str
    """
    BUFF_LEN = 10
    MIN_MATCHES_THRESH = 1

    def __init__(self, verbose=True, debug=False):
        self.detector = cv2.AKAZE_create()
        self.bridge = CvBridge()
        self.is_verbose = verbose
        self.is_debug = debug
        self.is_init = False
        self.img_buff = None
        self.polar_map_buff = None

        #### PUBLISHERS ####
        self.image_pub = rospy.Publisher('/features_debug', Image,
                                         queue_size=1)
        self.tf_pub = tf.TransformBroadcaster()

        #### Init Bearings ####
        # TODO(aldoteran): use ping result for this
        self.high_freq_bearings = np.linspace(35*np.pi/180.0, -35*np.pi/180.0, 256)

        #### Get sonar extrinsics ####
        self.tf_listener = tf.TransformListener(cache_time=rospy.Duration(20))
        self.tf_listener.waitForTransform("/slam/dead_reckoning/base_pose",
                                          "/slam/dead_reckoning/sonar_pose",
                                          rospy.Time(), rospy.Duration(3.0))
        trans, rot = self.tf_listener.lookupTransform('/slam/dead_reckoning/base_pose',
                                                      '/slam/dead_reckoning/sonar_pose',
                                                      rospy.Time(0))
        pose = tf.transformations.quaternion_matrix(rot)
        pose[:-1, -1] = np.asarray(trans)
        self.sonar_extrinsics = pose

        #### SUBSCRIBERS ####
        rospy.Subscriber('/sonar_image',
                         Image, self._image_cb)
        rospy.Subscriber('/simple_ping_result',
                         Float64MultiArray, self._ping_cb)

    def _image_cb(self, msg):
        # Get image
        img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        img = cv2.flip(img, 1)
        # Get pose
        try:
            trans, rot = self.tf_listener.lookupTransform('/world',
                                                        # '/slam/optimized/sonar_pose',
                                                        '/fiducials/sonar_pose',
                                                        rospy.Time(0))
            pose = tf.transformations.quaternion_matrix(rot)
            pose[:-1, -1] = np.asarray(trans)
        except:
            pose = self.sonar_extrinsics

        self.img_buff = (img, msg.header.stamp, msg.header.seq, pose)

    def _ping_cb(self, msg):
        ping_result = msg.data
        resolution = ping_result[8]
        rows = ping_result[9]
        range_max = rows*resolution

        ranges = np.linspace(0, range_max, rows)
        bearing_mesh, range_mesh = np.meshgrid(self.high_freq_bearings, ranges)

        self.polar_map_buff = (bearing_mesh, range_mesh, ping_result[1],
                               ping_result[0])

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
        try:
            for m,n in matches:
                if m.distance < 0.9*n.distance:
                    good_matches.append([m])
        except:
            pass

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
        T_Xb = np.linalg.inv(imgs[0][-1]).dot(imgs[1][-1])

        self.landmarks = [Landmark(features[0], features[1], maps[0], maps[1],
                                   match[0].queryIdx, match[0].trainIdx, T_Xb)
                          for match in features[2]]

        try:
            lucky = random.sample(range(len(self.landmarks)), 10)
            self.landmarks = np.asarray(self.landmarks)[lucky].tolist()
        except:
            pass

        return self.landmarks

def main():
    import time
    rospy.init_node('landmark_detection')
    detector = LandmarkDetector(verbose=True)
    rospy.loginfo("Initializing landmark detection.")
    rospy.sleep(3.0)
    # initialize with two images first
    img2 = detector.img_buff
    while not rospy.is_shutdown():
        tic = time.time()
        img1 = img2
        img2 = detector.img_buff
        detector.extract_n_match([img1, img2])
        toc = time.time()
        rospy.loginfo("Extracted and matched features in {} seconds".format(toc-tic))

if __name__ == '__main__':
    main()
