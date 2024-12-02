import numpy as np
import pybullet as p
import time
import json
# Colors
RED = (1, 0, 0, 1)
BLACK = (0, 0, 0, 1)
BLUE = (0.1, 0.2, 0.8, 1)
WHITE = (1, 1, 1, 1)
GREEN = (0.2, 0.8, 0.1, 1)
# Simulation params
SIM_DT = 1./100.
EPSILON = 1e-4
DECIMATION = 1
# Control params
P_GAIN = 1.0
D_GAIN = 0.0
# Render params
DISABLE_DRAWING = False
MAX_POINTS = 40000
RNG = np.random.default_rng()
# Maze params
wall_urdf = "assets/maze_wall.urdf"
corner_urdf = "assets/maze_corner.urdf"
boundary_urdf = "assets/maze_boundary.urdf"
wall_length = 1.0
wall_height = 0.5
maze_size = 10

def wrap_to_pi(angle):
    """
    Args:
        angle (float or 1D array): Angle in rad

    Returns:
        float or 1D array: Angle in [-pi, pi]
    """
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

def draw_sphere_markers(configs, color, marker_size=3.):
    """
    Args:
        configs (list): (N, dim_config) in task space [X, Y, ...]
        color (list): (3, ) in RGB
        marker_size (float): marker size
    """
    if DISABLE_DRAWING:
        return

    num_point = len(configs)
    if num_point == 0:
        return

    config_array = np.array(configs) # (N, dim_config) config should start with x and y
    if num_point > MAX_POINTS:
        print(f"Too many points ({num_point}) to draw, will downsample to {MAX_POINTS}.")
        num_point = MAX_POINTS
        config_array = RNG.choice(config_array, num_point, replace=False)

    point_positions = np.zeros((num_point, 3))
    point_positions[:, :2] = config_array[:, :2] # x, y
    point_positions[:, -1] = 0.1 # z
    p.addUserDebugPoints(pointPositions=point_positions, 
                         pointColorsRGB=[color[:3] for _ in range(num_point)], 
                         pointSize=marker_size, 
                         lifeTime=0)

def extract_path(node, keep_node=False) -> list:
    """
    Args:
        node (Node): Last node whose parents line a path
        keep_node (bool, optional): Return node path. Defaults to False.

    Returns:
        list: Path from earliest parent to node
    """
    path = []
    while node is not None:
        if not keep_node:
            # Extract config path
            path.append([*node.config])
        else:
            # Extract node path
            path.append(node)
        node = node.parent
    path.reverse()

    return path

def extract_bidirectional_path(node_from_start, node_to_goal, keep_node=False, steered_states=None):
    """
    Args:
        node_from_start (Node): Last node whose earliest parent is the start node
        node_to_goal (Node): Last node whose earliest parent is the goal node
        keep_node (bool, optional): Return node path. Defaults to False.
        steered_states (list): Steered states between node_from_start and node_to_goal

    Returns:
        list: Path from start to goal
    """
    path_from_start = extract_path(node_from_start, keep_node)
    path_to_goal = extract_path(node_to_goal, keep_node)
    path_to_goal.reverse()
    if keep_node:
        for i in range(1, len(path_to_goal)):
            path_to_goal[i].parent = path_to_goal[i-1]
        
        assert path_from_start[-1] == path_to_goal[0]

    path = []
    path.extend(path_from_start)
    if not keep_node and (steered_states is not None):
        path.extend(steered_states)

    if len(path_to_goal) > 1:
        path.extend(path_to_goal[1:])

    return path

def execute_trajectory(robot_id, joint_indices, path, draw=False):
    """
    Args:
        robot_id (int): Unique id from pybullet
        joint_indices (list): Actuated joints
        path (list): Path in config space
    """
    if not path:
        print("No path to execute!")
        return

    num_joints = len(joint_indices)
    command = np.zeros(num_joints * 2,)
    p_gains = [P_GAIN for _ in range(num_joints)]
    d_gains = [D_GAIN for _ in range(num_joints)]
    for config in path:
        command[:] = 0
        command[:len(config)] = config
        p.setJointMotorControlArray(robot_id, joint_indices, 
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPositions=command[:num_joints], 
                                    # targetVelocities=command[num_joints:], # TODO check if vel is necessary
                                    positionGains=p_gains,
                                    # velocityGains=d_gains,
                                    )

        for _ in range(DECIMATION):
            p.stepSimulation()
            time.sleep(SIM_DT)

        if draw:
            link_state = p.getLinkState(robot_id, joint_indices[-1])
            p.addUserDebugPoints(pointPositions=[link_state[0]], 
                                 pointColorsRGB=[RED[:3]], 
                                 pointSize=3., 
                                 lifeTime=0)

    # Debug info of PD pose tracking
    joint_states = p.getJointStates(robot_id, joint_indices)
    joint_positions = []
    for state in joint_states:
        joint_positions.append(state[0]) # (pos, vel, force, motor torque)
    print_opts  = np.get_printoptions()
    np.set_printoptions(precision=5, suppress=True)
    print("Target pos:\n", np.array(path[-1][:num_joints]))
    print("Achieved pos:\n", np.array(joint_positions))
    np.set_printoptions(**print_opts)

def get_collision_fn(robot_id, joint_indices):
    """
    Args:
        robot_id (int): Unique id of robot
        joint_indices (array_like): Indices of actuated joints
    """

    def collision_fn(config):
        """
        Args:
            config (array_like): Robot config

        Returns:
            bool: True if collided, else False
        """
        # Set joints to config
        set_joint_positions(robot_id, joint_indices, config)

        p.performCollisionDetection()
        contact_points = p.getContactPoints(robot_id)
        if len(contact_points) > 0:
            return True

        return False

    return collision_fn

def set_joint_positions(robot_id, joint_indices, positions):
    """Don't call this between simulation steps.

    Args:
        robot_id (int): Unique id of robot
        joint_indices (array_like): Indices of actuated joints
        positions (array_like): Positions w.r.t. joints to be set
    """
    # Make sure i-th joint is corresponding to i-th pos
    # and len(joints) <= len(positions)
    for joint, value in zip(joint_indices, positions):
        p.resetJointState(robot_id, joint, value)

def parse_actuated_joints(robot_id, return_limits=False):
    """
    Args:
        robot_id (int): unique id of robot
        return_limits (bool): parse revolute joint limits

    Returns:
        list: actuated joints
    """
    actuated_joints = []
    joint_limits = []
    num_joint = p.getNumJoints(robot_id)
    for idx in range(num_joint):
        joint_info_list = p.getJointInfo(robot_id, idx)
        if joint_info_list[2] != p.JOINT_FIXED:
            actuated_joints.append(joint_info_list[0])
        
        if return_limits and joint_info_list[2] == p.JOINT_REVOLUTE:
            joint_limits.append([*joint_info_list[8:10]])

    if return_limits:
        assert len(joint_limits) == len(actuated_joints)
        return joint_limits, actuated_joints

    return actuated_joints

def calculate_path_quality(path):
    """
    Args:
        path (list): planned path

    Returns:
        float: path quality based on the traveled distance in config space
    """
    path_cost = 0.0
    for idx in range(1, len(path)):
        prev_pt = path[idx-1]
        curr_pt = path[idx]
        dtheta = np.abs(curr_pt[2] - prev_pt[2])
        path_cost += np.sqrt((curr_pt[0] - prev_pt[0])**2 + 
                             (curr_pt[1] - prev_pt[1])**2 + 
                             min(dtheta, 2*np.pi-dtheta)**2)
    return path_cost

def add_wall_segment(x, y, value):
    """
    Args:
        x (int): x index
        y (int): y index
        value (int): wall type, 1 for horizontal, 2 for vertical, 
        3-6 for corners in clockwise direction
    """
    position = [x * wall_length - maze_size/2., 
                y * wall_length - maze_size/2., 
                wall_height/2.]
    if value == 1:  # Horizontal wall
        # Place horizontal wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF(wall_urdf, basePosition=position, 
                   baseOrientation=orientation, useFixedBase=True)
    elif value == 2:  # Vertical wall
        # Place vertical wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 1.5707963268])
        p.loadURDF(wall_urdf, basePosition=position, 
                   baseOrientation=orientation, useFixedBase=True)
    elif value == 3:  # Top-left corner
        # Place top-left corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 3.1415926536])
        p.loadURDF(corner_urdf, basePosition=position, 
                   baseOrientation=orientation, useFixedBase=True)
    elif value == 4:  # Top-right corner
        # Place top-right corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 1.5707963268])
        p.loadURDF(corner_urdf, basePosition=position, 
                   baseOrientation=orientation, useFixedBase=True)
    elif value == 5:  # Bottom-left corner
        # Place bottom-left corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF(corner_urdf, basePosition=position, 
                   baseOrientation=orientation, useFixedBase=True)
    elif value == 6:  # Bottom-right corner
        # Place bottom-right corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, -1.5707963268])
        p.loadURDF(corner_urdf, basePosition=position, 
                   baseOrientation=orientation, useFixedBase=True)

def build_maze(filename="maze_layout.json"):
    """
    Args:
        filename (str, optional): Defaults to "maze_layout.json".
    """
    # Define the maze layout as a 2D array
    with open(filename, 'r') as file:
        data = json.load(file)
        maze_layout = np.array(data['maze'])

    # Currently only support loading a 10x10 maze
    assert maze_layout.shape == (maze_size, maze_size)

    # Load maze boundary
    p.loadURDF(boundary_urdf, 
               basePosition=[-maze_size/2., -maze_size/2., wall_height/2.], 
               baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
               useFixedBase=True)

    # Loop through the maze_layout and add wall segments
    for y, row in enumerate(maze_layout):
        for x, obs_type in enumerate(row):
            if obs_type != 0:
                add_wall_segment(x, maze_size - y, obs_type)
    
    print(f"Loaded maze layout\n", maze_layout)

def get_transform_from_quat(quaternion):
    transform = np.zeros((4, 4))
    transform[:3, :3] = np.array(p.getMatrixFromQuaternion(quaternion)).reshape((3, 3))
    transform[-1, -1] = 1.
    return transform

def get_com_pose(body_id, link_idx):
    if link_idx==-1:
        return p.getBasePositionAndOrientation(body_id)
    
    link_state = p.getLinkState(body_id, link_idx)
    return np.array(link_state[0]), np.array(link_state[1]) # linkWorldPosition, linkWorldOrientation
