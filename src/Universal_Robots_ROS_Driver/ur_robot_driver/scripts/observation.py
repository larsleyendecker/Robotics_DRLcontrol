#!/usr/bin/env python2
import os
import numpy
import time

import rospy
from geometry_msgs.msg import WrenchStamped
from realsense2_camera.msg import EstimatedPose
from realsense2_camera.msg import CustomVector

def normalize_rad(angles):
    angles = numpy.array(angles)
    angles = angles % (2*numpy.pi)
    angles = (angles + 2*numpy.pi) % (2*numpy.pi)
    for i in range(len(angles)):
        if (angles[i] > numpy.pi):
            angles[i] -= 2*numpy.pi
    return angles

class Observator():
    def __init__(self):
        '''Observator constructor'''
        rospy.init_node("Observator", anonymous=True)
        rospy.Subscriber("/rotation_matrix", CustomVector, self.rotmat_callback)
        rospy.Subscriber("/ft300_force_torque", WrenchStamped, self.ft_callback)
        rospy.Subscriber("/pose_estimation", EstimatedPose, self.pose_callback)
        self.observation_publisher = rospy.Publisher("/observation", CustomVector, queue_size=1)
        self.rate = rospy.Rate(30)

    def ft_callback(self, data):
        '''ROS Callback function for the force-torque data'''
        self.ft_values = numpy.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])  

    def rotmat_callback(self, data):
        self.x_mat_flat = numpy.array(data.data)

    def pose_callback(self, data):
        '''ROS Callback function for the pose_estimation'''
        self.pose = numpy.array([data.tx, data.ty, data.tz, data.rx, data.ry, data.rz])

    def publish_observations(self):
        '''Function that computes and publishes the observations given the pose, rotation matrix and force torque data'''
        msg = CustomVector()
        x_mat = self.x_mat_flat.copy().reshape((3,3))
        pos = self.pose.copy()[:3]
        rpy = self.pose.copy()[3:]
        msg.data = numpy.concatenate([
            x_mat.dot(pos), x_mat.dot(normalize_rad(rpy)), self.ft_values.copy()
        ])
        self.observation_publisher.publish(msg)


if __name__ == "__main__":
    # Instantiate an Observator, wait for the topics to come up and start publishing
    observator = Observator()
    time.sleep(3) 
    while not rospy.is_shutdown():
        observator.publish_observations()
        observator.rate.sleep()