#!/usr/bin/env python
import os
import numpy
import numpy as np
import time
import mujoco_py
from gym.envs.robotics import rotations

import rospy
from geometry_msgs.msg import WrenchStamped
from realsense2_camera.msg import EstimatedPose
from realsense2_camera.msg import CustomVector
from sensor_msgs.msg import JointState

import cv2
import json
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType

import rospy
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera.msg import EstimatedPose

#HOME = os.getenv("HOME")
#MODEL_PATH = os.path.join(*[HOME, "DRL_AI4RoMoCo", "code", "environment", "UR10", "ur10_heg.xml"])

#############################################
def normalize_rad(angles):
    '''Normalizing euler angles'''
    angles = numpy.array(angles)
    # 1. reduce the angles
    angles = angles % (2*numpy.pi)
    # 2. force it to be the positive remainder, so that 0 <= angle < 360 (2pi)
    angles = (angles + 2*numpy.pi) % (2*numpy.pi)
    # 3. force into the minimum absolute value residue class, so that -180 (-pi) < angle <= 180 (pi)
    for i in range(len(angles)):
        if (angles[i] > numpy.pi):
            angles[i] -= 2*numpy.pi
    return angles

class Observator():

    def __init__(self):
        '''Observator constructor'''
        self.HOME = os.getenv("HOME")
        self.MODEL_PATH = os.path.join(*[self.HOME, "DRL_AI4RoMoCo", "code", "environment", "UR10_single", "ur10_heg.xml"])
        self.q_init = numpy.array([0, -1.3, 2.1, -0.80, 1.5708, 0.0])
        self.goal = numpy.array([0.69423743, -0.83110109,  1.17388998, -1.57161506,  0.02185773, -3.14102438])

        self.charuco_config = {
            "squaresX" : 5,
            "squaresY" : 5,
            "squareLength" : 0.0115,
            "markerLength" : 0.008,
            "dictionary" : cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
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

        # Initializing previous vectors
        self.previous_translation_vector = numpy.zeros(3,)
        self.previous_rotation_vector = numpy.zeros(3,)
        self.previous_tvec = numpy.zeros(3,)
        self.previous_rvec = numpy.zeros(3,)

        rospy.init_node("Observator", anonymous=True)
        rospy.Subscriber("/ft300_force_torque", WrenchStamped, self.ft_callback)

        rospy.Subscriber("/joint_states", JointState, self.q_callback)

        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.calibration_callback)

        self.observation_publisher = rospy.Publisher("/observation", CustomVector, queue_size=1)
        self.rate = rospy.Rate(30)
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(self.MODEL_PATH))
        self.set_state(self.q_init)

        ####################################################

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
            command.tx = -tz + self.correction["tx"] + 0.0016 + 0.002
            command.ty = tx + self.correction["ty"] - 0.0007 #- 0.015 ###
            command.tz = ty + self.correction["tz"] + 0.001
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

        self.final_pose = numpy.concatenate([tx, ty, tz])
        #self.pose_publisher.publish(command)

        #####################################################

    def set_state(self, qpos):
        '''Sets the state of the simulated model given the joint angles qpos'''
        #assert qpos.shape == (model.nq,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                    old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def q_callback(self, data):
        '''Callback function for receiving Joint States'''
        q = numpy.array(data.position)[[2,1,0,3,4,5]]
        self.set_state(q)
        self.xmat = self.sim.data.get_body_xmat("gripper_dummy_heg")

    def ft_callback(self, data):
        '''ROS Callback function for the force-torque data'''
        self.ft_values = (-1)* numpy.array([1*data.wrench.force.x, 1*data.wrench.force.y, 1*data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])  

    def rotmat_callback(self, data):
        self.x_mat_flat = numpy.array(data.data)

    #def pose_callback(self, data):
    #    '''ROS Callback function for the pose_estimation'''
    #    self.pose = numpy.array([data.tx, data.ty, data.tz, data.rx, data.ry, data.rz])

    def publish_observations(self):
        '''Function that computes and publishes the observations given the pose, rotation matrix and force torque data'''
        msg = CustomVector()
        
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        pos = self.final_pose
        rpy = normalize_rad(rotations.mat2euler(x_mat))

        msg.data = numpy.concatenate([
            x_mat.dot(pos), x_mat.dot(normalize_rad(rpy-self.goal[3:])), self.ft_values.copy()
        ])

        self.observation_publisher.publish(msg)


if __name__ == "__main__":
    # Instantiate an Observator, wait for the topics to come up and start publishing
    observator = Observator()
    time.sleep(3) 
    while not rospy.is_shutdown():
        observator.publish_observations()
        observator.rate.sleep()