import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import time
import pybullet as p
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("assets/robot.urdf",startPos, startOrientation)

# p.changeDynamics(robotId, -1, lateralFriction=0.0)

num_joint = p.getNumJoints(robotId)
print(f"Num joint: {num_joint}")

for idx in range(num_joint):
    joint_info_list = p.getJointInfo(robotId, idx)

    print(f"Index: {joint_info_list[0]} | Type: {joint_info_list[2]} | Link name: {joint_info_list[12]} \t| Joint name: {joint_info_list[1]}")

active_joints = [0, 1, 2]

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    p.setJointMotorControlArray(robotId, [0, 1, 2], 
                                controlMode=p.POSITION_CONTROL, 
                                targetPositions=[1, 1, 1], 
                                targetVelocities=[0.0, 0.0, 0.0],
                                positionGains=[0.1, 0.1, 0.1],
                                velocityGains=[0.9, 0.9, 0.9])
    p.setJointMotorControl2(robotId, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity=10, force=10)
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()
