#!/usr/bin/python

import rospy
import tf
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

if __name__ == "__main__":
    rospy.init_node("tf_to_path_publisher", anonymous=True)
    path_pub = rospy.Publisher("fiducials_path", Path, queue_size=10)
    odom_pub = rospy.Publisher("fiducials_odom", Odometry, queue_size=10)
    listener = tf.TransformListener()
    path = Path()
    path.header.frame_id = 'world'
    odom = Odometry()
    odom.header.frame_id = 'world'
    odom.child_frame_id = 'fiducials/base_pose'
    rate = rospy.Rate(50)

    # Run publisher
    while not rospy.is_shutdown():
        try:
            trans, rot = listener.lookupTransform('/world', '/fiducials/base_pose',
                                                  rospy.Time(0))
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = trans[0]
            pose.pose.position.y = trans[1]
            pose.pose.position.z = trans[2]
            pose.pose.orientation.x = rot[0]
            pose.pose.orientation.y = rot[1]
            pose.pose.orientation.z = rot[2]
            pose.pose.orientation.w = rot[3]
            pose.header.stamp = rospy.Time.now()
            path.poses.append(pose)


            odom.header.stamp = rospy.Time.now()
            odom.pose.pose.position = pose.pose.position
            odom.pose.pose.orientation = pose.pose.orientation
        except:
            continue

        path_pub.publish(path)
        odom_pub.publish(odom)
        rate.sleep()
