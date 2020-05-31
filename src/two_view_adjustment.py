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
    rospy.sleep(1.0)
    rospy.init_node('two_view_sonar_adjustment')
    rospy.loginfo("Initializing Two-View sonar bundle adjustment.")
    detector = LandmarkDetector(verbose=is_verbose)
    adjuster = BundleAdjuster(verbose=is_verbose, iters=6)
    rospy.sleep(0.5)
    # wait for buffer to fill up
    img2 = detector.img_buff
    maps_img2 = detector.cart_map_buff
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        tic = time.time()
        # if len(detector.img_buff) == 0 or len(detector.cart_map_buff) == 0:
            # continue
        img1 = img2
        maps_img1 = maps_img2
        img2 = detector.img_buff
        maps_img2 = detector.cart_map_buff
        try:
            assert img1[1] == maps_img1[3] and img2[1] == maps_img2[3]
        except:
            continue
        features = detector.extract_n_match([img1, img2])
        landmarks = detector.generate_landmarks([img1, img2], features,
                                                [maps_img1, maps_img2])
        if landmarks == None or len(landmarks) == 0:
            continue
        adjuster.compute_constraint(landmarks, Xa=img1[-1], Xb=img2[-1])
        toc = time.time()
        rospy.loginfo("Computed constraint in {} seconds".format(toc-tic))
        rate.sleep()

if __name__ == '__main__':
    main()

