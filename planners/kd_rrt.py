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

    def steer(self, state_from, state_to, num_points=None, method="direct"):
        if method == "direct":
            return self.steer_direct(state_from, state_to)
        elif method == "indirect":
            return self.steer_bvp(state_from, state_to)
        else:
            raise NotImplementedError()

    def steer_direct(self, state_from, state_to):
        """
            Solve trajectory optimization via direct trajectory optimization
            to connect two states within a small region
        """
        tf = 5 # sec
        num_points = int(tf/SIM_DT)
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
    
    def steer_bvp(self, state_from, state_to):
        """
            Solve the ODE system with boundary conditions derived from 
            continuous optimal control via Pontryagin's Minimum Principle (PMP)
            to connect two states within a small region.
        """
        # x: time
        # y: state + costate
        dim_state = self._robot.dim_state
        dim_ctrl = self._robot.dim_ctrl
        n = dim_state * 2 # state + costate

        ya_tgt = np.zeros((n*2,))
        ya_tgt[:n] = state_from
        yb_tgt = np.zeros((n*2,))
        yb_tgt[:n] = state_to
        ctrl_limits = self._robot._ctrl_bounds[:, [0, -1]] # (dim_ctrl, 2) [lb, ub]

        def get_u(p, lb, ub):
            """ Optimal control trajectory derived from PMP
                g(x, u) = u^2
            """
            u = 0
            if p >= 2:
                u = lb
            elif p <= -2:
                u = ub
            else:
                u = -0.5 * p
            
            return u

        def fun(x, y):
            """ The derived ODE system
                n state equations
                n costate equations
                2n boundary conditions on states
                dy / dx = f(x, y)
            """
            # x: (m, )
            # y: (n, m)
            m = x.shape[0]
            state = y[:dim_state]   # (dim_state, num_points)
            costate = y[dim_state:] # (dim_state, num_points)
            control = np.zeros((dim_ctrl, m)) # (dim_ctrl, num_points)
            
            for idx in range(m):
                for idx_ctrl in range(dim_ctrl):
                    control[idx_ctrl, idx] = get_u(costate[dim_ctrl+idx_ctrl, idx], 
                                                   ctrl_limits[idx_ctrl, 0]*0.01,
                                                   ctrl_limits[idx_ctrl, 1]*0.01)

            dydx = np.zeros((n, m)) # (n, 1)
            dydx[:dim_state] = self._robot.ss_continuous(state, control)
            dydx[dim_state:] = - self._robot._A.T @ costate
            return dydx

        def bc(ya, yb):
            """ Boundary conditions (2n)
                bc(y(a), y(b)) = 0
            """
            # ya, yb: (n, )
            return np.array([ya[0] - ya_tgt[0], ya[1] - ya_tgt[1], wrap_to_pi(ya[2] - ya_tgt[2]), # pose at a
                             ya[3] - ya_tgt[3], ya[4] - ya_tgt[4], ya[5] - ya_tgt[5], # vel at a
                             yb[0] - yb_tgt[0], yb[1] - yb_tgt[1], wrap_to_pi(yb[2] - yb_tgt[2]), # pose at b
                             yb[3] - yb_tgt[3], yb[4] - yb_tgt[4], yb[5] - yb_tgt[5], # vel at b
                             ])
        tf = 5 # sec
        num_points = int(tf/SIM_DT)
        x_mesh = np.linspace(0, tf, num_points) # (num_points, )
        y_init = np.linspace(ya_tgt, yb_tgt, num_points, axis=-1) # (n, num_points)
        res = solve_bvp(fun, bc, x_mesh, y_init, verbose=2, max_nodes=1e4, tol=0.001)
        print(f"BVP Success: {res.success} | {res.message}")
        res_states = res.y[:n].T # (num_points, dim_state)

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
                    print(f"[Error] Steered path collided...")
                    return []
                path.append(state)

            draw_sphere_markers(path, RED)
            draw_sphere_markers(self._configs, BLUE)
        
        return path
