#!/bin/env/python

import tf
import numpy as np

import rospy
from sensor_msgs.msg import Imu

class BiasCompensator:

    def __init__(self):

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.theta_z = 0.0
        self.dt = 1.0/25.0
        self.time = 0.0
        self.is_init = False
        self.is_compensated = False
        self.orientation = np.ones(3)
        self.imu_pub = rospy.Publisher('/imu/filtered_data', Imu, queue_size=10)
        rospy.Subscriber("/imu/data", Imu, self._imu_callback)

    def _imu_callback(self, msg):
        if not self.is_init:
            _,self.orientation,_ = self._get_rotation(msg)
            self.is_init = True

        if not self.is_compensated:
            accel = np.array([[msg.linear_acceleration.x],
                              [msg.linear_acceleration.y],
                              [msg.linear_acceleration.z]])
            accel = self.orientation.dot(accel)
            accel[2,0] -= 9.818
            vel = accel * self.dt
            self.vel_x += vel[0]
            self.vel_y += vel[1]
            self.vel_z += vel[2]
            self.theta_x += msg.angular_velocity.x * self.dt
            self.theta_y += msg.angular_velocity.y * self.dt
            self.theta_z += msg.angular_velocity.z * self.dt
            self.time += self.dt
        else:
            imu = msg
            rotation, base_rotation, quat = self._get_rotation(msg)
            # Compensate accelerations
            accel = np.array([[msg.linear_acceleration.x],
                              [msg.linear_acceleration.y],
                              [msg.linear_acceleration.z]])
            accel = self.orientation.dot(accel)
            # accel -= self.accel_bias
            accel[2,0] -= 9.818
            accel = self.orientation.transpose().dot(accel)
            # Compensate angular velocities
            omega = np.array([[msg.angular_velocity.x],
                              [msg.angular_velocity.y],
                              [msg.angular_velocity.z]])
            # omega -= self.omega_bias
            # Replace values and publish
            imu.orientation.x = quat[0]
            imu.orientation.y = quat[1]
            imu.orientation.z = quat[2]
            imu.orientation.w = quat[3]
            imu.linear_acceleration.x = accel[0,0]
            imu.linear_acceleration.y = accel[1,0]
            imu.linear_acceleration.z = accel[2,0]
            imu.angular_velocity.x = omega[0,0]
            imu.angular_velocity.y = omega[1,0]
            imu.angular_velocity.z = omega[2,0]
            self.imu_pub.publish(imu)

    def _get_rotation(self, msg):
        quat = [0.,0.,0.,0.]
        quat[0] = msg.orientation.x
        quat[1] = msg.orientation.y
        quat[2] = msg.orientation.z
        quat[3] = msg.orientation.w
        orientation = tf.transformations.quaternion_matrix(quat)
        euler = tf.transformations.euler_from_quaternion(quat)
        base_rotation = tf.transformations.euler_matrix(euler[0]+np.pi,
                                                        -euler[1],
                                                        0.0)
        quat = tf.transformations.quaternion_from_euler(euler[0]+np.pi,
                                                       -euler[1],
                                                        0.0)

        return orientation[:-1,:-1], base_rotation[:-1,:-1], quat

    def normalize_quat(self,quat):
        quat = np.asarray(quat)
        length = np.linalg.norm(quat)

        return (quat/length).tolist()

    def estimate_bias(self):
        print("Velocities")
        print("X: {}, Y: {}, Z: {}".format(self.vel_x,
                                           self.vel_y,
                                           self.vel_z))
        print("Angles")
        print("Roll: {}, Pitch: {}, Yaw: {}".format(self.theta_x,
                                                    self.theta_y,
                                                    self.theta_z))
        print("Time: {}".format(self.time))

        acc_x_bias = self.dt * self.vel_x/self.time
        acc_y_bias = self.dt * self.vel_y/self.time
        acc_z_bias = self.dt * self.vel_z/self.time
        self.accel_bias = np.array([acc_x_bias,
                                    acc_y_bias,
                                    acc_z_bias])
        # self.accel_bias = np.array([[0.0048],
                                    # [0.009],
                                    # [0.0048]])
        omega_x_bias = self.dt * self.theta_x/self.time
        omega_y_bias = self.dt * self.theta_y/self.time
        omega_z_bias = self.dt * self.theta_z/self.time
        self.omega_bias = np.array([[omega_x_bias],
                                    [omega_y_bias],
                                    [omega_z_bias]])
        self.is_compensated = True

        print("--------------")
        print("Estimated Acceleration Bias")
        print("X: {}".format(acc_x_bias))
        print("Y: {}".format(acc_y_bias))
        print("Z: {}".format(acc_z_bias))
        print("Estimated Angular Velocity Bias")
        print("Roll: {}".format(omega_x_bias))
        print("Pitch: {}".format(omega_y_bias))
        print("Yaw: {}".format(omega_z_bias))

        rospy.logwarn("Bias estimated, compensating IMU measurements.")

def main():
    rospy.init_node("bias_self")
    rospy.sleep(1)
    rospy.logwarn("Identifying bias...")
    compensator = BiasCompensator()
    rate = rospy.Rate(25)
    time_now = rospy.Time.now()
    end_time = rospy.Time.from_sec(time_now.to_sec() + 3.0)
    while time_now <= end_time:
        time_now = rospy.Time.now()
        rate.sleep()
    compensator.estimate_bias()
    while not rospy.is_shutdown():
        rate.sleep()
        rospy.spin()


if __name__ == "__main__":
    main()
