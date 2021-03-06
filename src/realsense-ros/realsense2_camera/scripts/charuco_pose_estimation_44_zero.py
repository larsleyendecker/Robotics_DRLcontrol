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

class PoseEstimator:

    def __init__(self):
        #with open("charuco_config.json") as config_file:
        #    self.charuco_config = json.load(config_file)
        #self.charuco_config["dictionary"] = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
        self.charuco_config = {
            "squaresX" :3,
            "squaresY" : 3,
            "squareLength" : 0.018,
            "markerLength" : 0.014,
            "dictionary" : cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        }
        self.offset = {
            "tx" : 0.03,
            "ty" : -0.0275,
            "tz" : -0.05265,
            "rx" : 0,
            "ry" : 0,
            "rz" : 0
        }
        self.correction = {
            "tx" : 0.002,
            "ty" : -0.0016,
            "tz" : -0.0026,
            "rx" : 0,
            "ry" : 0,
            "rz" : 0,
        }

        self.cos_transformation = True
        
        self.filter = True
        self.N = 5
        self.tx_raws = list(numpy.zeros(self.N))
        self.ty_raws = list(numpy.zeros(self.N))
        self.tz_raws = list(numpy.zeros(self.N))
        self.rx_raws = list(numpy.zeros(self.N))
        self.ry_raws = list(numpy.zeros(self.N))
        self.rz_raws = list(numpy.zeros(self.N))

        self.tvec_filtered = numpy.zeros((3,1))
        self.rvec_filtered = numpy.zeros((3,1))

        self.correct_rx = False
        self.rx_correction = -0.18

        self.bridge = CvBridge()
        self.charuco_board = self.create_charuco_board(self.charuco_config)
        self.params = cv2.aruco.DetectorParameters_create()
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("camera/color/camera_info", CameraInfo, self.calibration_callback)
        self.pose_publisher = rospy.Publisher("/pose_estimation", EstimatedPose, queue_size=1)
        self.rate = rospy.Rate(30)

        # Initializing previous vectors
        self.previous_translation_vector = numpy.zeros(3,)
        self.previous_rotation_vector = numpy.zeros(3,)
        self.previous_tvec = numpy.zeros(3,)
        self.previous_rvec = numpy.zeros(3,)

    def create_charuco_board(self, config):
        '''Creates and returns a charuco board with the given configuration'''
        return cv2.aruco.CharucoBoard_create(config["squaresX"], config["squaresY"], config["squareLength"], config["markerLength"], config["dictionary"])

    def charuco_pose_estimation(self, cv_image):
        '''Does the pose estimation using the charuco board (chessboard + aruco marker) in the given configuration'''
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.charuco_config["dictionary"], parameters=self.params)
        if ids is not None:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, cv_image, self.charuco_board, minMarkers=1)
            cv2.aruco.drawDetectedCornersCharuco(cv_image, charuco_corners, charuco_ids)
            retval, rotation_vector, translation_vector = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.camera_matrix, self.distortion_coefficients, None, None)
            if retval == True:
                cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.distortion_coefficients, rotation_vector, translation_vector, 0.03)
                cv2.imshow("Charuco", cv_image)
                #self.previous_translation_vector = translation_vector
                #self.previous_rotation_vector = rotation_vector
                return translation_vector, rotation_vector
            else:
                cv2.imshow("Charuco", cv_image)
                #return numpy.zeros(3), numpy.zeros(3)
                return None, None #numpy.zeros(3), numpy.zeros(3) # None, None #self.previous_translation_vector, self.previous_rotation_vector
        else:
            cv2.imshow("Charuco", cv_image)
            #return numpy.zeros(3), numpy.zeros(3)
            return None, None #numpy.zeros(3), numpy.zeros(3) #None, None #self.previous_translation_vector, self.previous_rotation_vector

    def image_callback(self, ros_image):
        '''Callback function for the subscription of the ROS topic /camera/color/image_raw (sensor_msgs Image)'''
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        tvec, rvec = self.charuco_pose_estimation(cv_image)
        if tvec is None:
            self.publish_pose(self.previous_tvec, self.previous_rvec)
        else:
            self.previous_tvec = tvec
            self.previous_rvec = rvec
            self.publish_pose(tvec, rvec)
        cv2.waitKey(1)

    def calibration_callback(self, data):
        '''Sets the calibration parameters - camera_matrix and distortion_coefficients - by reading the ROS topic /camera/color/cameraInfo'''
        self.camera_matrix = numpy.array(data.K).reshape(3,3)
        self.distortion_coefficients = numpy.array(data.D).reshape(5,)

    def filter_pose(self, tvec, rvec):
        '''Running Mean Filter for estimated pose'''
        if self.filter:
            tvec_raw = [self.tx_raws, self.ty_raws, self.tz_raws]
            rvec_raw = [self.rx_raws, self.ry_raws, self.rz_raws]
            i = 0
            j = 0 

            for t in tvec_raw:
                t.append(tvec[i])
                t.pop(0)
                self.tvec_filtered[i] = numpy.mean(t)
                i += 1
            for r in rvec_raw:
                r.append(rvec[j])
                r.pop(0)
                self.rvec_filtered[j] = numpy.mean(r)
                j += 1
            return self.tvec_filtered, self.rvec_filtered
        else:
            return tvec, rvec

    def publish_pose(self, tvec, rvec):
        '''Publishes the estimated pose from the charuco board to a ROS topic'''

        tvecF, rvecF = self.filter_pose(tvec, rvec)
        command = EstimatedPose()
        tx = tvecF[0] + self.offset["tx"] #- 0.18
        ty = tvecF[1] + self.offset["ty"]
        tz = tvecF[2] + self.offset["tz"]
        rx = rvecF[0] + self.offset["rx"]
        ry = rvecF[1] + self.offset["ry"]
        rz = rvecF[2] + self.offset["rz"] + self.rx_correction

        if self.cos_transformation:
            command.tx = -tz + self.correction["tx"]
            command.ty = tx + self.correction["ty"]
            command.tz = ty + self.correction["tz"]
            command.rx = -rz + self.correction["rx"]
            command.ry = -rx + self.correction["ry"]
            command.rz = -ry + self.correction["rz"]
        else:
            command.tx = tx
            command.ty = ty
            command.tz = tz
            command.rx = rx
            command.ry = ry
            command.rz = rz

        self.pose_publisher.publish(command)

if __name__ == "__main__":
    rospy.init_node("charuco_estimator", anonymous=True)
    PoseEstimator()
    rospy.spin()