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
        self.MODEL_PATH = os.path.join(*[self.HOME, "DRL_AI4RoMoCo", "code", "environment", "UR10", "ur10_heg.xml"])
        self.q_init = numpy.array([0, -1.3, 2.1, -0.80, 1.5708, 0.0])
        self.goal = numpy.array([0.69423743, -0.83110109,  1.17388998, -1.57161506,  0.02185773, -3.14102438])

        rospy.init_node("Observator", anonymous=True)
        rospy.Subscriber("/ft300_force_torque", WrenchStamped, self.ft_callback)
        rospy.Subscriber("/pose_estimation", EstimatedPose, self.pose_callback)
        rospy.Subscriber("/joint_states", JointState, self.q_callback)
        self.observation_publisher = rospy.Publisher("/observation", CustomVector, queue_size=1)
        self.rate = rospy.Rate(30)
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(self.MODEL_PATH))
        self.set_state(self.q_init)

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
        self.ft_values = numpy.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])  

    def rotmat_callback(self, data):
        self.x_mat_flat = numpy.array(data.data)

    def pose_callback(self, data):
        '''ROS Callback function for the pose_estimation'''
        self.pose = numpy.array([data.tx, data.ty, data.tz, data.rx, data.ry, data.rz])

    def publish_observations(self):
        '''Function that computes and publishes the observations given the pose, rotation matrix and force torque data'''
        msg = CustomVector()
        
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        pos = self.pose.copy()[:3]
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