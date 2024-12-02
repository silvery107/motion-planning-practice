import time
import numpy as np
import casadi as ca
from .rrt import RRT, Node
from scipy.integrate import solve_bvp
from utils import *


class KdRRT(RRT):
    """Kinodynamic Rapidly Exploring Random Tree
    """
    def __init__(self, 
                 config_space, 
                 collision_fn, 
                 *, 
                 robot_model, 
                 num_primitives=100,
                 uniform_primitive=False,
                 goal_bias=0.1, 
                 connect_thresh = 0.5,
                 **kwargs):

        super().__init__(config_space, collision_fn, goal_bias=goal_bias)
        self._robot = robot_model
        self._num_primitives = num_primitives
        self._uniform_primitive = uniform_primitive
        self._connect_thresh = connect_thresh

    def extend_node(self, near_node: Node, rand_node: Node) -> Node:
        """
            u* = min_{u_i} dist(f(x_near, u_i), x_rand) for u_i in U
            x_new = f(x_near, u*)
        """
        primitives = self._robot.sample_control(self._num_primitives, 
                                               uniform=self._uniform_primitive) # (dim_ctrl, N)
        new_states = self._robot.ss_discrete(near_node.config[:, None], 
                                             primitives) # (N, dim_state)
        valid_mask = np.logical_and(np.less_equal(new_states, 
                                                  self._config_space[:, 1]).all(1), 
                                    np.greater_equal(new_states, 
                                                    self._config_space[:, 0]).all(1)) # (N, )

        new_states = new_states[valid_mask]
        if len(new_states) < 1:
            # Can't expand anymore
            return near_node

        diff = new_states - rand_node.config # (N, dim_state)
        dist = np.linalg.norm(diff @ self._dist_weights_mat, axis=1) # (N, )
        min_idx = np.argmin(dist)
        new_state = new_states[min_idx]

        return Node(new_state)

    def get_rand_node(self, target_node: Node = None) -> Node:
        rand_node = super().get_rand_node(target_node)
        # Set target velocities to zero for each waypoints
        rand_node.config[3:] = 0
        return rand_node

    def is_connected(self, node_from, node_to, thresh=0.5):
        # Test a circle region considering only x and y distance
        if np.linalg.norm(node_from.config[:2] - node_to.config[:2]) < thresh:
            return True

        return False

    def steer(self, state_from, state_to, num_points=500):
        """
            Solve trajectory optimization to connect two states within a small region
        """
        dim_state = self._robot.dim_state
        dim_ctrl = self._robot.dim_ctrl
        # Def dynamics
        dt = self._robot._dt
        x = ca.MX.sym('x', dim_state)
        u = ca.MX.sym('u', dim_ctrl)
        x_next = ca.MX.zeros(dim_state)
        x_next[3:] = x[3:] + u * dt
        x_next[:3] = x[:3] + x_next[3:] * dt
        F = ca.Function('F', [x, u], [x_next])
        # Build a Multiple Shooting TO with Casadi Opti stack
        opti = ca.Opti()
        # Opt variables
        X = opti.variable(dim_state, num_points+1)
        U = opti.variable(dim_ctrl, num_points)
        # Boundary constraints
        opti.subject_to(X[:, 0] == state_from)
        opti.subject_to(X[:, -1] == state_to)
        # Dynamics constraints
        for i in range(num_points):
            opti.subject_to(X[:, i+1] == F(X[:, i], U[:, i]))
        # Path constraints
        ctrl_ub = self._robot._ctrl_bounds[:, -1]
        opti.subject_to(opti.bounded(-ctrl_ub, U, ctrl_ub))
        opti.subject_to(ca.sum1((X[:2, :] - state_to[:2])**2) <= self._connect_thresh**2)
        # Objective
        opti.minimize(ca.sumsqr(U))
        # Initial guess
        opti.set_initial(X, np.linspace(state_from, state_to, num_points+1).T)
        # Solver
        options = dict()
        options["print_time"] = True
        options["ipopt"] = {"print_level": 0, "tol": 1e-6}
        opti.solver("ipopt", options)
        sol = None
        try:
            sol = opti.solve()
        except:
            opti.debug.show_infeasibilities()
            # print(opti.debug.value(U))
            # import pdb
            # pdb.set_trace()
            print("Steering failed...")

        if sol is not None:
            return sol.value(X).T # (num_points+1, dim_state)

        return []
    
    
    def plan_path(self, start_config, goal_config):
        path = []
        # Initialize start and goal node
        start_node = Node(np.array([*start_config]))
        goal_node = Node(np.array([*goal_config]))
        self.append(start_node)
        self.set_dist_weights([1, 1, 1, 0.0001, 0.0001, 0.0001])
        path_found = False
        start_time = time.time()
        while not path_found:
            # Try connect RRT to goal node
            self.connect_to_target(goal_node)

            # Check if goal node is connected
            if self.is_connected(self.back(), goal_node):
                steered_states = self.steer(self.back().config, goal_node.config)
                path_found = True
                break

        if path_found:
            print("Path Found!!!")
            print(f"Time elapsed: {time.time() - start_time:.5f}")
            path = extract_path(self.back())
            # BVP solutions
            for state in steered_states:
                if self._collision_fn(state):
                    print(f"[Error] Steered path collided...")
                    return []
                path.append(state)

            draw_sphere_markers(path, RED)
            draw_sphere_markers(self._configs, BLUE)
        
        return path
