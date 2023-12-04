import time
import numpy as np
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
                 step_size=0.05, 
                 **kwargs):

        super().__init__(config_space, collision_fn, goal_bias=goal_bias, step_size=step_size)
        self._robot = robot_model
        self._num_primitives = num_primitives
        self._uniform_primitive = uniform_primitive

    def extand_node(self, near_node: Node, rand_node: Node) -> Node:
        """
            u* = min_{u_i} dist(f(x_near, u_i), x_rand) for u_i in U
            x_new = f(x_near, u*)
        """
        primitives = self._robot.sample_control(self._num_primitives, 
                                               uniform=self._uniform_primitive) # (dim_ctrl, N)
        new_states = self._robot.ss_discrete(near_node.config[:, None], primitives) # (N, dim_state)
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

    def is_connected(self, node_from, node_to):
        # Test a circle region considering only x and y distance
        if np.linalg.norm(node_from.config[:2] - node_to.config[:2]) < 0.5:
            return True

        return False

    def steer(self, state_from, state_to, num_points=40):
        """
            Solve BVP to connect two states within a small region
        """
        # y: state + control
        # x: time
        n = self._robot.dim_state + self._robot.dim_ctrl

        ya_tgt = np.zeros((n,))
        ya_tgt[:self._robot.dim_state] = state_from
        yb_tgt = np.zeros((n,))
        yb_tgt[:self._robot.dim_state] = state_to

        def fun(x, y):
            # x: (m, )
            # y: (n, )
            state = y[:self._robot.dim_state] # (dim_state, num_points)
            control = y[self._robot.dim_state:] # (dim_ctrl, num_points)
            dydx = np.zeros((n, x.shape[0]))
            dydx[:self._robot.dim_state] = self._robot.ss_continuous(state, control)
            return dydx # (n, m)

        def bc(ya, yb):
            # ya, yb: (n, )
            return np.array([ya[0] - ya_tgt[0], ya[1] - ya_tgt[1], ya[2] - ya_tgt[2], # pose at a
                            #  ya[3] - ya_tgt[3], ya[4] - ya_tgt[4], ya[5] - ya_tgt[5], # vel at a
                             yb[0] - yb_tgt[0], yb[1] - yb_tgt[1], yb[2] - yb_tgt[2], # pose at b
                             yb[3] - yb_tgt[3], yb[4] - yb_tgt[4], yb[5] - yb_tgt[5], # vel at b
                             ])

        x_mesh = np.linspace(0, 1, num_points) # (num_points, )
        y_init = np.linspace(ya_tgt, yb_tgt, num_points, axis=-1) # (dim_state+dim_ctrl, num_points)
        res = solve_bvp(fun, bc, x_mesh, y_init, verbose=1, max_nodes=200, tol=0.001)
        print(f"BVP Success: {res.success} | {res.message}")
        res_states = res.y[:self._robot.dim_state].T # (num_points, dim_state)

        if res.success:
            return res_states

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
                    print(f"Steered path collided...")
                    break
                path.append(state)

            draw_sphere_markers(path, RED)
            draw_sphere_markers(self._configs, BLUE)
        
        return path
