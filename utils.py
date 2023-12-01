import numpy as np
import pybullet as p
import time
# Colors
RED = (1, 0, 0, 1)
BLACK = (0, 0, 0, 1)
BLUE = (0, 0, 1, 1)
WHITE = (1, 1, 1, 1)
GREEN = (0, 1, 0, 1)
# Simulation params
SIM_DT = 1./240.
EPSILON = 1e-4
DECIMATION = 4
# Control params
P_GAIN = 0.8
D_GAIN = 0.0
# Render params
MAX_POINTS = 35000
RNG = np.random.default_rng()

def wrap_to_pi(angle):
    """
    Args:
        angle (float or 1D array): Angle in rad

    Returns:
        float or 1D array: Angle in [-pi, pi]
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

def draw_sphere_markers(configs, color, marker_size=3.):
    """
    Args:
        configs (list): (N, dim_config) in task space [X, Y, ...]
        color (list): (3, ) in RGB
        marker_size (float): marker size
    """
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

def execute_trajectory(robot_id, joint_indices, path):
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
                                    targetPositions=command[:3], 
                                    # targetVelocities=command[3:], # TODO check if vel is necessary
                                    positionGains=p_gains,
                                    # velocityGains=d_gains,
                                    )

        for _ in range(DECIMATION):
            p.stepSimulation()
            time.sleep(SIM_DT)

    # Debug info of PD pose tracking
    body_state = p.getLinkState(robot_id, 3)
    pos = body_state[0]
    orn = body_state[1]
    print_opts  = np.get_printoptions()
    np.set_printoptions(precision=5, suppress=True)
    print("Target pose:\n", np.array(path[-1][:3]))
    print("Robot pose:\n", np.array([*pos[:2], p.getEulerFromQuaternion(orn)[-1]]))
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
