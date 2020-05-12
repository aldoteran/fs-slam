#!/usr/bin/python2.7
"""
ROS node to execute the two view sonar bundle adjustment as
described in the paper "Degeneracy-Aware Imaging Sonar
Simultaneous Localization and Mapping" by Westman and Kaess.
"""
import rospy
import time
from BundleAdjuster import BundleAdjuster
from LandmarkDetector import LandmarkDetector

__license__ = "MIT"
__author__ = "Aldo Teran, Antonio Teran"
__author_email__ = "aldot@kth.se, teran@mit.edu"
__status__ = "Development"

is_verbose = rospy.get_param("verbose")

def main():
    rospy.init_node('two_view_sonar_adjustment')
    detector = LandmarkDetector(verbose=is_verbose)
    rospy.sleep(2.0)
    adjuster = BundleAdjuster(verbose=is_verbose)
    rospy.loginfo("Initializing Two-View sonar bundle adjustment.")
    rospy.sleep(2.0)
    # initialize with two images first
    if len(detector.img_buff) >= 2:
        img1 = detector.img_buff.pop(0)
        img2 = detector.img_buff.pop(0)
        maps_img1 = detector.cart_map_buff.pop(0)
        maps_img2 = detector.cart_map_buff.pop(0)
        features = detector.extract_n_match([img1, img2])
        if not features == None:
            landmarks = detector.generate_landmarks([img1, img2], features,
                                                    [maps_img1, maps_img2])
            adjuster.compute_constraint(landmarks)
    while not rospy.is_shutdown():
        tic = time.time()
        if len(detector.img_buff) >= 1 & len(detector.cart_map_buff) >=1:
            # drop only first image
            img1 = img2
            img2 = detector.img_buff.pop(0)
            maps_img1 = maps_img2
            maps_img2 = detector.cart_map_buff.pop(0)
            features = detector.extract_n_match([img1, img2])
            if not features == None:
                landmarks = detector.generate_landmarks([img1, img2], features,
                                                        [maps_img1, maps_img2])
                adjuster.compute_constraint(landmarks)
            toc = time.time()
            rospy.loginfo("Computed constraint in {} seconds".format(toc-tic))



if __name__ == '__main__':
    main()

