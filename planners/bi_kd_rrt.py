import time
import numpy as np
from .bi_rrt import BiRRT
from .kd_rrt import KdRRT
from .rrt import Node
from utils import *


class BiKdRRT(BiRRT):
    def __init__(self, 
                 config_space, 
                 collision_fn, 
                 *, 
                 robot_model, 
                 steer_thresh=0.1,
                 steer_pts=20,
                 num_primitives=100,
                 uniform_primitive=False,
                 goal_bias=0.1, 
                 **kwargs):
        self.rrt_start = KdRRT(config_space, 
                               collision_fn, 
                               robot_model=robot_model, 
                               num_primitives=num_primitives,
                               uniform_primitive=uniform_primitive,
                               goal_bias=goal_bias)
        self.rrt_goal = KdRRT(config_space, 
                              collision_fn, 
                              robot_model=robot_model, 
                              num_primitives=num_primitives,
                              uniform_primitive=uniform_primitive,
                              goal_bias=goal_bias)
        self._collision_fn = collision_fn
        self._steer_threshold = steer_thresh
        self._steer_points = steer_pts


    def plan_path(self, start_config, goal_config):
        path = []
        # Initialize start and goal node
        start_node = Node(np.array([*start_config]))
        goal_node = Node(np.array([*goal_config]))
        self.rrt_start.append(start_node)
        self.rrt_goal.append(goal_node)
        self.rrt_start.set_dist_weights([1, 1, 1, 0.0001, 0.0001, 0.0001])
        self.rrt_goal.set_dist_weights([1, 1, 1, 0.0001, 0.0001, 0.0001])
        path_found = False
        start_time = time.time()
        while not path_found:
            # Pick last node from RRT Goal
            node_from_rrt_goal = self.rrt_goal.back()
            # Try connect RRT Start to RRT Goal
            self.rrt_start.connect_to_target(node_from_rrt_goal)

            # Check if both trees are connected
            if self.rrt_start.is_connected(self.rrt_start.back(), 
                                           self.rrt_goal.back(), 
                                           self._steer_threshold):
                steered_states = self.rrt_start.steer(self.rrt_start.back().config, 
                                                      self.rrt_goal.back().config, 
                                                      self._steer_points)
                path_found = True
                break

            # Pick last node from RRT Start
            node_from_rrt_start = self.rrt_start.back()
            # Try connect RRT Goal to RRT Start
            self.rrt_goal.connect_to_target(node_from_rrt_start)

            # Check if both trees are connected
            if self.rrt_goal.is_connected(self.rrt_start.back(), 
                                          self.rrt_goal.back(), 
                                          self._steer_threshold):
                steered_states = self.rrt_goal.steer(self.rrt_start.back().config, 
                                                     self.rrt_goal.back().config, 
                                                     self._steer_points)
                path_found = True
                break

        # Check BVP solutions
        for state in steered_states:
            if self._collision_fn(state):
                print(f"[Error] Steered path collided...")
                return []
            path.append(state)

        print("Path Found!!!")
        print(f"Time elapsed: {time.time() - start_time:.5f}")

        path = extract_bidirectional_path(self.rrt_start.back(), 
                                          self.rrt_goal.back(), 
                                          steered_states=steered_states)
        draw_sphere_markers(path, RED)
        draw_sphere_markers(self.rrt_start._configs, BLUE)
        draw_sphere_markers(self.rrt_goal._configs, GREEN)
        
        return path
