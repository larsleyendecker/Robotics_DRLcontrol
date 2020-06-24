#!/usr/bin/env python2
import os
import numpy

import rospy
from realsense2_camera.msg import EstimatedPose, CustomVector

def pose_callback(data):
    '''ROS Callback function for the pose_estimation'''
    global pose
    pose = numpy.array([data.tx, data.ty, data.tz, data.rx, data.ry, data.rz])
    observation_publisher.publish(pose)

def pose_publish(posex):
    msg = CustomVector()
    msg.data = posex
    observation_publisher.publish(msg)

if __name__ == "__main__":
    rospy.init_node("pose_tester", anonymous=True)
    observation_publisher = rospy.Publisher("/xPoseTest", CustomVector, queue_size=1)
    rospy.Subscriber("/pose_estimation", EstimatedPose, pose_callback)
    rospy.spin()
    '''
    while not rospy.is_shutdown():
        pose_publish(pose)
        rate.sleep()
    '''