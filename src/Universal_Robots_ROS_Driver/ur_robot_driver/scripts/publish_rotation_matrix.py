#!/usr/bin/env python
import os
import gym
import numpy
import mujoco_py

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from realsense2_camera.msg import CustomVector

HOME = os.getenv("HOME")
MODEL_PATH = os.path.join(*[HOME, "DRL_AI4RoMoCo", "code", "environment", "UR10", "ur10_heg.xml"])
q_init = numpy.array([0, -1.3, 2.1, -0.80, 1.5708, 0.0])

def set_state(qpos):
    '''Sets the state of the simulated model given the joint angles qpos'''
    #assert qpos.shape == (model.nq,)
    old_state = sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                    old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()

def q_callback(data):
    '''Callback function for receiving Joint States'''
    q = numpy.array(data.position)[[2,1,0,3,4,5]]
    set_state(q)
    xmat = sim.data.get_body_xmat("gripper_dummy_heg")
    publish_xmat(xmat)

def publish_xmat(xmat):
    '''Message which includes the rotation matrix'''
    #message = Float64MultiArray()
    message = CustomVector()
    message.data = xmat.reshape((9,))
    xmat_publisher.publish(message)

if __name__ == "__main__":
    rospy.init_node("forward_kinematizer", anonymous=True)
    sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(MODEL_PATH))
    set_state(q_init)
    rospy.Subscriber("/joint_states", JointState, q_callback)
    xmat_publisher = rospy.Publisher("/rotation_matrix", CustomVector, queue_size=1)

    rospy.spin()