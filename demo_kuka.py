import numpy as np
import pybullet as p
from argparse import ArgumentParser
import pybullet_data

from planners import *
from utils import *
import utils

parser = ArgumentParser()
parser.add_argument("--algo", choices=["RRT", "BiRRT"], default="BiRRT")
args = parser.parse_args()

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    utils.DISABLE_DRAWING = True
    
    # Iitialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(SIM_DT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=1.3,
                                 cameraYaw=135,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[0, 0., 0.3])

    # Load world plane and robot
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    plane_id = p.loadURDF("plane.urdf")
    start_pos = [0, 0, 0]
    start_euler = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler(start_euler)
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation, useFixedBase=True)

    add_wall_segment(5.2, 5.2, 6)

    # Parse active DoFs from robot urdf
    joint_limits, actuated_joints = parse_actuated_joints(robot_id, return_limits=True)
    joint_polarities = np.ones(len(actuated_joints))
    joint_polarities[3] *= -1 #! make sure polarity of each joint is correct
    print(f"Actuated joint indices: {actuated_joints}")

    # Define robot state space
    state_space = np.array(joint_limits).reshape((-1, 2))
    print("State Spaces of the 7-DOF Arm:\n", state_space)
    # Collision detection callback
    collision_fn = get_collision_fn(robot_id, actuated_joints)

    # Initialize start and goal positions in [x, y, z]
    start_pos = [0.0, 0.5, 0.4]
    goal_pos = [0.5, 0.0, 0.4]
    p.addUserDebugPoints(pointPositions=[start_pos], 
                         pointColorsRGB=[BLACK[:3]], 
                         pointSize=15, 
                         lifeTime=0)
    p.addUserDebugPoints(pointPositions=[goal_pos], 
                         pointColorsRGB=[GREEN[:3]], 
                         pointSize=15, 
                         lifeTime=0)

    ik_solver = IIK(robot_id, actuated_joints, joint_limits, joint_polarities, alpha=0.001, beta=0.1)
    config_guess = np.zeros(len(actuated_joints))
    start_config = ik_solver.calculate_ik(start_pos, config_guess)
    goal_config = ik_solver.calculate_ik(goal_pos, config_guess)

    assert not collision_fn(goal_config) and not collision_fn(start_config)
    path = []

    # RRT parameters
    step_size = 0.02 # unit vec in config space
    goal_bias = 0.1 # 5% ~ 10%
    # Bi-directional Kinodynamic RRT parameters
    steer_threshold = 0.1
    steer_points = 40


    ### Run planner container
    print(f"Start path planning with {args.algo}...")
    algo_container = eval(args.algo)(state_space, 
                                     collision_fn, 
                                     goal_bias=goal_bias, 
                                     step_size=step_size, 
                                     steer_thresh=steer_threshold,
                                     steer_pts=steer_points,
                                     )
    path = algo_container.plan_path(start_config, goal_config)
    ###

    # Execute planned path
    set_joint_positions(robot_id, actuated_joints, start_config)
    print("Start executing planned path...")
    execute_trajectory(robot_id, actuated_joints, path, draw=True)

    input("Done!")
    p.disconnect()
