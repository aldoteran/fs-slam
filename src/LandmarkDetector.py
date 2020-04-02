#!/usr/bin/python2.7
"""
Landmark detection node for the FS sonar SLAM pipeline,
following the methods used in the paper.
"""
import rospy
import tf
import cv2
import timeit
import ros_numpy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseWithCovariance

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

    :param range_map: Mapping for the image ranges
    :type range_map: list [floats]

    :param swath_map: Mapping for the image swaths
    :type swath_map: list [floats]
    """
    def __init__(self, keypts1, keypts2, key1, key2, range_map, swath_map):
        self.pixels_img1 = tuple(np.round(keypts1[key1].pt).astype(int))
        self.pixels_img2 = tuple(np.round(keypts2[key2].pt).astype(int))
        # These are the measurements of m_i from img1(X_a) and img2(X_b)
        #TODO(aldoteran): should these be gaussian random vars?
        self.cart_img1 = np.array([[range_map[self.pixels_img1[0]]],
                                [swath_map[self.pixels_img1[1]]]])
        self.cart_img2 = np.array([[range_map[self.pixels_img2[0]]],
                                [swath_map[self.pixels_img2[1]]]])
        self.polar_img1 = self._cart_to_polar(self.cart_img1)
        self.polar_img2 = self._cart_to_polar(self.cart_img2)
        # init elevation and phi to 0
        self.phi = 0.0
        self.elevation = 0.0
        #TODO: do we need a position in the world/map/odom frame?

    def update_phi(self, phi):
        # Updates phi and elevation attributes (wrt X_a/img1)
        self.phi = phi
        self.elevation = self.cart_img1[0] * np.sin(phi)

    def _cart_to_polar(self, cart):
        return np.array([[np.arctan2(cart[0],cart[1])],
                         [np.sqrt(cart[0]**2 + cart[1]**2)]])


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

    def __init__(self, features='AKAZE'):
        self.detector = self.feat_dict[features]
        self.bridge = CvBridge()
        self.img_buff = []
        self.range_buff = []
        self.pts_buff = []
        self.range_map = None
        self.swath_map = None
        self.is_init = False

        #### SUBSCRIPTIONS ####
        #TODO(aldoteran): Maybe using a sonar img handler is more modular
        # self.img_handler = SimSonarHandler()
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber('/rexrov/depth/image_raw_raw_sonar',
                         Image, self._image_cb)
        rospy.Subscriber("/rexrov/points", PointCloud2, self._pts_cb)

        #### PUBLISHERS ####
        self.pose_pub = rospy.Publisher('/sonar_constraint',
                                        PoseWithCovariance,
                                        queue_size=1)
        self.image_pub = rospy.Publisher('/features_debug',
                                         Image, queue_size=1)

    def _image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        img = np.uint8(255 * (img / np.max(img)))
        stamp = msg.header.stamp
        self.img_buff.append((img,stamp))

    def _pts_cb(self, msg):
        """
        Creates two vectors using the information in the pointcloud.
        The combination of those vectors will represent the X and Y
        cartesian coordinates of each pixel in the sonar image with
        the same timestamp.
        NOTE: PointCloud2 message comes with ROS camera frame convention:
              Z -> Forward, X -> Right, Y -> Down.
        """
        if self.is_init:
            return

        pcl = ros_numpy.point_cloud2.pointcloud2_to_array(msg, squeeze=False)
        self.range_buff.append(pcl)

        ranges = pcl['z'][0]
        swaths = pcl['x'][0]
        # remove NaNs
        ranges = ranges[~np.isnan(ranges)]
        swaths = swaths[~np.isnan(swaths)]

        # get max ranges and resolutions
        range_max = max(ranges)
        swath_max = max(swaths)
        swath_min = min(swaths)
        #TODO(aldoteran): make img shape available globally
        range_res = range_max / 400 #img.shape[0]
        swath_res = (abs(swath_min) + swath_max) / 478 #img.shape[1]
        self.range_map = np.arange(0.0, range_max, range_res)
        self.swath_map = np.arange(swath_min, swath_max, swath_res)

        self.is_init = True


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
        toc = timeit.timeit()

        rospy.logwarn("keypoints: {}, descriptors: {}".format(len(keypts1), descript1.shape))
        rospy.logwarn("keypoints: {}, descriptors: {}".format(len(keypts2), descript2.shape))
        rospy.logwarn("Computed in: {} seconds".format(toc-tic))

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
        img_msg = self.bridge.cv2_to_imgmsg(feat_img, encoding="passthrough")
        self.image_pub.publish(img_msg)

        return [keypts1, keypts2, good_matches]

    def generate_landmarks(self, imgs, features):
        """
        Generates a list of Landmark instances and updates each
        one with its optima elevation angle following Eq.(17) in the paper.

        :params imgs: pair of images to extract and match features from
        :type imgs: list of tuples [(img1,stamp), (img2,stamp)]

        :params features: list of keypoints and matches for the features
                          in both images
        :type features: list [keypts_img1, keypts_img2, [cv2.DMatch,...,N]]
        """
        self.landmarks = [Landmark(features[0], features[1],
                                   match[0].queryIdx, match[0].trainIdx,
                                   self.range_map, self.swath_map)
                          for match in features[2]]
        #TODO(aldoteran): for m_i in landmarks find optimal phi.

def main():
    rospy.init_node('feature_extraction')
    detector = LandmarkDetector(features='AKAZE')
    rospy.sleep(1)
    # initialize with two first images
    if len(detector.img_buff) >= 2:
        img1 = detector.img_buff.pop(0)
        img2 = detector.img_buff.pop(0)
        features = detector.extact_n_match([img1, img2])
        detector.generate_landmarks([img1, img2], features)
        import pdb
        pdb.set_trace()

    while not rospy.is_shutdown():
        if len(detector.img_buff) >= 1:
            # drop only first image
            img1 = img2
            img2 = detector.img_buff.pop(0)
            detector.extact_n_match([img1, img2])

if __name__ == '__main__':
    main()











