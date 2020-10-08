#!/usr/bin/env python

import os
import numpy
import cv2
import json
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType

import rospy
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera.msg import EstimatedPose

class ImageViewer:

    def __init__(self):

        self.bridge = CvBridge()

        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("camera/color/camera_info", CameraInfo, self.calibration_callback)
        self.pose_publisher = rospy.Publisher("/pose_estimation", EstimatedPose, queue_size=1)
        self.rate = rospy.Rate(30)

    def image_callback(self, ros_image):
        '''Callback function for the subscription of the ROS topic /camera/color/image_raw (sensor_msgs Image)'''
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        cv2.imshow("RealSense Check", cv_image)
        cv2.waitKey(1)

    def calibration_callback(self, data):
        '''Sets the calibration parameters - camera_matrix and distortion_coefficients - by reading the ROS topic /camera/color/cameraInfo'''
        self.camera_matrix = numpy.array(data.K).reshape(3,3)
        self.distortion_coefficients = numpy.array(data.D).reshape(5,)

if __name__ == "__main__":
    rospy.init_node("image_viewer", anonymous=True)
    ImageViewer()
    rospy.spin()