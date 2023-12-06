import numpy as np
import pybullet as p
from argparse import ArgumentParser
import pybullet_data

from common import Hovercraft
from planners import *
from utils import *

parser = ArgumentParser()
parser.add_argument("--algo", choices=["RRT", "BiRRT", "KdRRT", "Astar", "BiKdRRT"], default="KdRRT")
args = parser.parse_args()

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    
    # Iitialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(SIM_DT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load world plane and robot
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    plane_id = p.loadURDF("plane.urdf")
    start_pos = [0, 0, 0]
    start_euler = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler(start_euler)
    robot_id = p.loadURDF("assets/robot.urdf", start_pos, start_orientation, useFixedBase=True)

    # Build the maze
    build_maze("maze_layout.json")

    # Parse active DoFs from robot urdf
    actuated_joints = parse_actuated_joints(robot_id)
    print(f"Actuated joint indices: {actuated_joints}")

    # Collision detection callback
    collision_fn = get_collision_fn(robot_id, actuated_joints)

    # Initialize start and goal config in [x, y, theta, dx, dy, dtheta]
    start_config = start_pos[:2] + [start_euler[-1]] + [0., 0., 0.]
    goal_config = [-2., -3., np.pi, 0., 0., 0.]
    assert not collision_fn(goal_config) and not collision_fn(start_config)
    draw_sphere_markers([[start_config[0], start_config[1], 0.1]], BLACK, 10)
    draw_sphere_markers([[goal_config[0], goal_config[1], 0.1]], GREEN, 10)
    path = []

    # RRT parameters
    step_size = 0.02 # unit vec in config space
    goal_bias = 0.1 # 5% ~ 10%
    # Kinodynamic RRT parameters
    num_primitives = 26
    uniform_primitive = False
    robot_model = Hovercraft(SIM_DT)
    # Bi-directional Kinodynamic RRT parameters
    steer_threshold = 0.1
    steer_points = 40
    # Astar parameters
    connectivity = 8

    # Define robot state space
    search_dim = 3 if args.algo=="RRT" or args.algo=="Astar" else 6
    state_space = np.array([-10, 10,
                            -10, 10,
                            -2*np.pi, 2*np.pi,
                            -1, 1,
                            -1, 1,
                            -1, 1
                            ]).reshape((-1, 2))
    print("State Spaces in [x, y, theta, dx, dy, dtheta]:\n", state_space[:search_dim])

    ### Run planner container
    print(f"Start path planning with {args.algo}...")
    algo_container = eval(args.algo)(state_space[:search_dim], 
                                     collision_fn, 
                                     goal_bias=goal_bias, 
                                     step_size=step_size, 
                                     connectivity=connectivity,
                                     robot_model=robot_model,
                                     num_primitives=num_primitives,
                                     uniform_primitive=uniform_primitive,
                                     steer_thresh=steer_threshold,
                                     steer_pts=steer_points,
                                     )
    path = algo_container.plan_path(start_config[:search_dim], goal_config[:search_dim])
    ###

    # Execute planned path
    path_cost = calculate_path_quality(path)
    print(f"Path quality: {path_cost:.5f}")
    set_joint_positions(robot_id, actuated_joints, start_config)
    print("Start executing planned path...")
    execute_trajectory(robot_id, actuated_joints, path)

    input("Done!")
    p.disconnect()
