import numpy as np

from utils import *


def get_ee_pos(robot_id, joint_indices, joint_vals=None):
    if joint_vals is not None:
        set_joint_positions(robot_id, joint_indices, joint_vals)

    pos, orn = get_com_pose(robot_id, joint_indices[-1])
    return pos

def get_joint_pos_and_axis(robot, joint_idx):
    # returns joint position and axis in the world frame
    j_info = p.getJointInfo(robot, joint_idx)

    jt_local_pos, jt_local_orn = j_info[14], j_info[15] # parentFramePos, parentFrameOrn
    H_L_J = get_transform_from_quat(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info[16]) # parentIndex
    H_W_L = get_transform_from_quat(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    # Joint position
    j_world_position = H_W_J[:3, 3]
    # Joint axis
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info[13]) # jointAxis
    joint_axis_world = np.dot(R_W_J, joint_axis_local)

    return j_world_position, joint_axis_world

def get_translation_jacobian(robot, joint_indices):
    J = np.zeros((3, len(joint_indices))) # (3, num_joint)
    # Get ee pos
    ee_pos = get_ee_pos(robot, joint_indices)
    for idx, joint_idx in enumerate(joint_indices):
        joint_pos_world, joint_axis_world = get_joint_pos_and_axis(robot, joint_idx) # (3, ), (3, )
        # Compute Jacobian for each joint
        J[:, idx] = np.cross(joint_axis_world, ee_pos - joint_pos_world)

    return J

def get_jacobian_pinv(J, lam=0.01):
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))
    return J_pinv


class IIK:
    """Iterative (Jacobian Pseudo-Inverse) Inverse Kinematics
    """
    def __init__(self, 
                 robot_id, 
                 joint_indices, 
                 joint_limits, 
                 joint_polarities, 
                 *, 
                 alpha=0.01, 
                 beta=1, 
                 max_iters=10000) -> None:

        self._robot_id = robot_id
        self._joint_indices = joint_indices
        self._config_space = np.array(joint_limits).reshape((-1, 2))
        self._joint_polarities = joint_polarities
        self._alpha = alpha
        self._beta = beta
        self._epsilon = 1e-3
        self._max_iters = max_iters

    def calculate_ik(self, x_target, q_guess=None):
        config_midpoint = (self._config_space[:, 1] + self._config_space[:, 0]) / 2.
        # Init states
        q_curr = q_guess if q_guess is not None else np.zeros((len(self._joint_indices)))
        iterations = 0
        while iterations < self._max_iters:
            iterations += 1
            x_current = get_ee_pos(self._robot_id, self._joint_indices, q_curr)
            x_dot = x_target - x_current
            x_error_norm = np.linalg.norm(x_dot)
            if x_error_norm < self._epsilon:
                break
            J = get_translation_jacobian(self._robot_id, self._joint_indices) # (3, num_joint)
            J *= self._joint_polarities
            J_inv = get_jacobian_pinv(J) # (num_joint, 3)
            q_dot = J_inv @ x_dot
            # Combining tasks using the null-space
            q_sec = config_midpoint - q_curr
            q_sec_norm = np.linalg.norm(q_sec)
            if q_sec_norm < EPSILON:
                q_sec[:] = 0.0
            else:
                q_sec /= np.linalg.norm(q_sec)
            q_dot += self._beta * (np.eye(J_inv.shape[0]) - J_inv @ J) @ q_sec
            q_dot_norm = np.linalg.norm(q_dot)
            if q_dot_norm > self._alpha:
                q_dot = self._alpha * (q_dot / q_dot_norm)

            q_curr += q_dot
            q_curr = np.clip(q_curr, self._config_space[:, 0], self._config_space[:, 1])
        
        print(f"IK solved for position {x_current} in {iterations} iterations with config:\n{q_curr}")
        return q_curr
