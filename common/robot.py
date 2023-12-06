import numpy as np


class Hovercraft:
    """
        State:   [x, y, theta, dx, dy, dtheta]
        Control: [ddx, ddy, ddtheta]
    """
    def __init__(self, dt) -> None:
        self.dim_state = 6
        self.dim_ctrl = 3

        self._dt = dt
        self._ctrl_scale = 1.
        self._A = np.zeros((self.dim_state, self.dim_state))
        self._A[:3, 3:] = np.eye(3)
        self._B = np.zeros((self.dim_state, self.dim_ctrl))
        self._B[3:] = np.eye(3)

        self._Ad = np.eye(self.dim_state) + self._A * self._dt
        self._Bd = self._B * self._dt

        self._ctrl_bounds = np.array([-1, 0, 1,
                                      -1, 0, 1,
                                      -1, 0, 1
                                     ]).reshape((self.dim_ctrl, 3)) * self._ctrl_scale

        control_primitives = set()
        for xdd in self._ctrl_bounds[0]:
            for ydd in self._ctrl_bounds[1]:
                for thetadd in self._ctrl_bounds[2]:
                    control_primitives.add((xdd, ydd, thetadd))

        control_primitives.remove((0, 0, 0))
        self._ctrl_primitives = np.array(list(control_primitives)).T * self._ctrl_scale # (3, 26)

    def ss_continuous(self, state, control):
        """Propagate continuous state space model
        \dot{x} = f(x, u)

        Args:
            state (ndarray): (dim_state, 1)
            control (ndarray): (dim_control, num_samples)

        Returns:
            ndarray: new states (num_samples, dim_state)
        """
        return self._A @ state + self._B @ control

    def ss_discrete(self, state, control):
        """Propagate discrete state space model
        x_{k+1} = f(x_k, u_k)

        Args:
            state (ndarray): (dim_state, 1)
            control (ndarray): (dim_control, num_samples)

        Returns:
            ndarray: new states (num_samples, dim_state)
        """
        new_states = np.zeros((self.dim_state, control.shape[1]))
        # Semi-Implicit Euler
        new_states[3:] = state[3:] + control * self._dt
        new_states[:3] = state[:3] + new_states[3:] * self._dt
        # Forward Euler
        # new_states = self._Ad @ state + self._Bd @ control # this works poorly in pybullet

        return new_states.T

    def sample_control(self, num_primitives=10, uniform=False):
        """sample control from discretized primitives or uniform distributions

        Args:
            num_primitives (int, optional): Defaults to 10.
            uniform (bool, optional): Defaults to False.

        Returns:
            ndarray: (dim_ctrl, num_primitives)
        """
        if uniform:
            return np.random.uniform(low=self._ctrl_bounds[:, :1], 
                                     high=self._ctrl_bounds[:, -1:], 
                                     size=(self.dim_ctrl, num_primitives))

        return np.random.permutation(self._ctrl_primitives.T)[:num_primitives].T
