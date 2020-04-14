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
from sensor_msgs.msg import Image#, PointCloud2
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray

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
    def __init__(self, keypts1, keypts2, map1, map2, key1, key2):
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

    def update_phi(self, phi):
        # Updates phi and elevation attributes (wrt X_a/img1)
        self.phi = phi
        self.elevation = self.cart_img1[0] * np.sin(phi)

    def project_coords(self, phi):
        """
        Returns the projection of the X, Y and Z coordinates
        given the elevation angle phi. Used for the search for
        the optimal phi. Initial coords are from img1, i.e., as
        seen from pose X_a.

        :param phi: elevation angle in radians
        :type phi: float

        :return: X, Y and Z coordinates of the landmark
        :rtype: np.array (3,1)
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
        # rospy.Subscriber("/rexrov/points", PointCloud2, self._pts_cb)

        #### PUBLISHERS ####
        self.pose_pub = rospy.Publisher('/sonar_constraint',
                                        PoseWithCovariance,
                                        queue_size=1)
        self.image_pub = rospy.Publisher('/features_debug', Image,
                                         queue_size=1)
        self.marker_pub = rospy.Publisher('/landmarks', MarkerArray,
                                          queue_size=1)

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
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        # self.cart_map_buff.append((cv2.dilate(img[:,:,0], kernel, iterations=2),
                                   # cv2.dilate(img[:,:,1], kernel, iterations=2),
                                   # cv2.dilate(img[:,:,2], kernel, iterations=2),
                                   # stamp))
        self.cart_map_buff.append((img[:,:,0],
                                   img[:,:,1],
                                   img[:,:,2],
                                   stamp))
        if len(self.cart_map_buff) >= self.BUFF_LEN:
            self.cart_map_buff.pop(0)

    def _pts_cb(self, msg):
        """
        [DEPRECATED] Creates two vectors using the information in the pointcloud.
        The combination of those vectors will represent the X and Y
        cartesian coordinates of each pixel in the sonar image with
        the same timestamp.
        NOTE: PointCloud2 message comes with ROS camera frame convention:
              Z -> Forward, X -> Right, Y -> Down.
        """
        # if self.is_init:
            # return

        pcl = ros_numpy.point_cloud2.pointcloud2_to_array(msg, squeeze=False)
        self.pcl_buff.append(pcl)

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
        range_map = np.arange(0.0, range_max, range_res)
        swath_map = np.arange(swath_min, swath_max, swath_res)

        self.sonar_frame = msg.header.frame_id
        stamp = msg.header.stamp
        self.cart_map_buff.append((range_map, swath_map, stamp))


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
        self.landmarks = [Landmark(features[0], features[1], maps[0], maps[1],
                                   match[0].queryIdx, match[0].trainIdx,)
                          for match in features[2]]

        # get poses and compute T_xb, following notation from the paper:
        trans_Xa, rot_Xa = self.tf_listener.lookupTransform('/rexrov/forward_sonar_optical_frame',
                                                         '/world', imgs[0][-1])
        trans_Xb, rot_Xb = self.tf_listener.lookupTransform('/rexrov/forward_sonar_optical_frame',
                                                         '/world', imgs[1][-1])
        Xa = tf.transformations.quaternion_matrix(rot_Xa)
        Xa[:-1, -1] = np.asarray(trans_Xa)
        Xb = tf.transformations.quaternion_matrix(rot_Xb)
        Xb[:-1, -1] = np.asarray(trans_Xb)
        T_Xb = np.dot(np.linalg.pinv(Xa), Xb)

        #TODO(aldoteran): import from config file
        # I believe that the VFOV from the depth camera is 63 deg, res=0.5deg
        phi_range = np.arange(-0.5497789, 0.5497789, 0.008726)
        sigma = np.float32(np.diag((0.01, 0.01)))
        for l in self.landmarks:
            if l.is_shit:
                continue
            phi_star = self._opt_phi_search(l, T_Xb, sigma, phi_range)
            l.update_phi(phi_star)

        # for debugging, publish landmark markers
        if self.is_verbose:
            landmarkers = MarkerArray()
            for i,l in enumerate(self.landmarks):
                if l.is_shit:
                    continue
                # Green ground truth markers
                landmarkers.markers.append(self._create_marker(l,i, imgs))
                # Red estimated markers
                landmarkers.markers.append(self._create_marker_est(l,i+len(self.landmarks), imgs))
            self.marker_pub.publish(landmarkers)

        return self.landamarks

    def _opt_phi_search(self, landmark, T_Xb, sigma, phi_range):
        """
        Search for optimal phi using the list of phis in phi_range.
        """
        rot_Xb = T_Xb[:-1,:-1]
        trans_Xb = np.expand_dims(T_Xb[:-1,-1],1)

        # project landmarks from X_a in X_b using all phis
        proj = [(landmark.project_coords(phi) - trans_Xb)
                 for phi in phi_range]
        proj = np.squeeze(np.asarray(proj))
        proj = np.dot(rot_Xb, proj.transpose())
        # compute error vector
        z_b = landmark.polar_img2
        error = np.zeros((proj.shape[1], 1), dtype=np.float32)
        for i in range(proj.shape[1]):
            # polar_proj = landmark.cart_to_polar(proj[:,i:i+1])
            # innov = polar_proj - np.float32(z_b)
            innov = np.float32(landmark.cart_to_polar(proj[:,i:i+1])) - np.float32(z_b)
            error[i] = innov.transpose().dot(np.linalg.inv(sigma)).dot(innov)
        phi_star_idx = np.argmin(error)

        return phi_range[phi_star_idx]

    def _create_marker(self, landmark, idx, imgs):
        # create markers to display in rviz for debugging
        marker = Marker()
        marker.header.stamp = imgs[0][1]
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
        marker.pose.position.x = landmark.cart_img2[1]
        marker.pose.position.y = landmark.cart_img2[2]
        marker.pose.position.z = landmark.cart_img2[0]

        return marker

    def _create_marker_est(self, landmark, idx, imgs):
        # create markers to display in rviz for debugging
        marker = Marker()
        marker.header.stamp = imgs[0][1]
        marker.header.frame_id = '/rexrov/forward_sonar_optical_frame'
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
        marker.pose.position.x = landmark.cart_img2[1]
        marker.pose.position.y = landmark.elevation
        marker.pose.position.z = landmark.cart_img2[0]

        return marker



def main():
    rospy.init_node('feature_extraction')
    detector = LandmarkDetector(features='AKAZE')
    rospy.sleep(2)
    # initialize with two first images
    if len(detector.img_buff) >= 2:
        img1 = detector.img_buff.pop(0)
        img2 = detector.img_buff.pop(0)
        maps_img1 = detector.cart_map_buff.pop(0)
        maps_img2 = detector.cart_map_buff.pop(0)
        features = detector.extact_n_match([img1, img2])
        detector.generate_landmarks([img1, img2], features,
                                    [maps_img1, maps_img2])
        import pdb
        pdb.set_trace()
    while not rospy.is_shutdown():
        if len(detector.img_buff) >= 1 & len(detector.cart_map_buff) >=1:
            # drop only first image
            img1 = img2
            img2 = detector.img_buff.pop(0)
            #TODO(aldoteran): assert stamps correspond btwn img and map
            maps_img1 = maps_img2
            maps_img2 = detector.cart_map_buff.pop(0)
            features = detector.extact_n_match([img1, img2])
            detector.generate_landmarks([img1, img2], features,
                                        [maps_img1, maps_img2])

if __name__ == '__main__':
    main()











