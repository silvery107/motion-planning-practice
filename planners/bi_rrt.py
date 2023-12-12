import time
import numpy as np
from .rrt import RRT, Node
from .smoothing import shortcut_smoothing
from utils import *

class BiRRT:
    """Bidirectional Rapidly Exploring Random Tree
    """
    def __init__(self, config_space, collision_fn, *, goal_bias=0.1, step_size=0.05, **kwargs):
        self.rrt_start = RRT(config_space, collision_fn, goal_bias=goal_bias, step_size=step_size)
        self.rrt_goal = RRT(config_space, collision_fn, goal_bias=goal_bias, step_size=step_size)
        self._collision_fn = collision_fn

    def plan_path(self, start_config, goal_config):
        path = []
        # Initialize start and goal node
        start_node = Node(np.array([*start_config]))
        goal_node = Node(np.array([*goal_config]))
        self.rrt_start.append(start_node)
        self.rrt_goal.append(goal_node)
        path_found = False
        start_time = time.time()
        while not path_found:
            # Pick last node from RRT Goal
            node_from_rrt_goal = self.rrt_goal.back()
            # Try connect RRT Start to RRT Goal
            self.rrt_start.connect_to_target(node_from_rrt_goal)

            # Check if both trees are connected
            if self.rrt_start.back() == self.rrt_goal.back():
                path_found = True
                break

            # Pick last node from RRT Start
            node_from_rrt_start = self.rrt_start.back()
            # Try connect RRT Goal to RRT Start
            self.rrt_goal.connect_to_target(node_from_rrt_start)

            # Check if both trees are connected
            if self.rrt_start.back() == self.rrt_goal.back():
                path_found = True
                break

        if path_found:
            print("Path Found!!!")
            print(f"Time elapsed: {time.time() - start_time:.5f}")
            path = extract_bidirectional_path(self.rrt_start.back(), self.rrt_goal.back())
            draw_sphere_markers(path, BLACK, 2.5)
            draw_sphere_markers(self.rrt_start._configs, BLUE)
            draw_sphere_markers(self.rrt_goal._configs, GREEN)
            # Path smoothing
            node_path = extract_bidirectional_path(self.rrt_start.back(), self.rrt_goal.back(), keep_node=True)
            node_path = shortcut_smoothing(self.rrt_start, node_path, self._collision_fn)
            path = extract_path(node_path[-1])
            draw_sphere_markers(path, RED)
        
        return path
