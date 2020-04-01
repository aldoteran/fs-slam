#!/usr/bin/python2.7
"""
Landmark detection node for the FS sonar SLAM pipeline.
"""
import rospy
import tf
import cv2
import timeit
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovariance

__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

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

        #### SUBSCRIPTIONS ####
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber('/rexrov/depth/image_raw_raw_sonar',
                         Image, self._image_cb)

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

    def extact_n_match(self, imgs):
        """
        Extract the features from the input image.

        :params img: image to extract the features from
        :type img: np.ndarray

        :return: extracted features
        :rtype: don't know yet
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

        return [imgs, (keypts1, keypts2), good_matches]

def main():
    rospy.init_node('feature_extraction')
    detector = LandmarkDetector(features='AKAZE')
    rospy.sleep(1)
    #TODO(aldoteran): drop only first image, keep second for next iteration
    if len(detector.img_buff) >= 2:
        img1 = detector.img_buff.pop(0)
        img2 = detector.img_buff.pop(0)
        detector.extact_n_match([img1, img2])
    while not rospy.is_shutdown():
        if len(detector.img_buff) >= 1:
            img1 = img2
            img2 = detector.img_buff.pop(0)
            detector.extact_n_match([img1, img2])

if __name__ == '__main__':
    main()











