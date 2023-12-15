import numpy as np
from common import AbcNode
from numpy import ndarray
from sklearn.neighbors import KDTree
from .smoothing import shortcut_smoothing
from utils import *


class Node(AbcNode):
    def __init__(self, config: ndarray, parent=None) -> None:
        super().__init__(config, parent)

class RRT:
    """Rapidly Exploring Random Tree
    """
    def __init__(self, 
                 config_space:ndarray,
                 collision_fn,
                 *,
                 goal_bias:float=0.1, 
                 step_size:float=0.05,
                 **kwargs):
        self._elements = []
        self._configs = []
        self._kdtree = None
        self._build_tree_flag = False
        self._build_tree_thresh = 0
        self._tree_increment = 5000
        self._collision_fn = collision_fn
        self._config_space = config_space # (N, 2)
        self._goal_bias = goal_bias
        self._step_size = step_size
        self._dist_weights = np.ones(len(config_space))
        self._dist_weights_mat = np.eye(len(config_space)) # squred weight matrix, multiplied before norm

    def append(self, node:Node):
        self._elements.append(node)
        self._configs.append(node.config)
        if len(self._configs) > self._build_tree_thresh and not self._build_tree_flag:
            self._build_tree_flag = True
            self._build_tree_thresh += self._tree_increment
    def back(self) -> Node:
        return self._elements[-1]

    def set_dist_weights(self, weights:list):
        assert len(weights) == self._config_space.shape[0]
        self._dist_weights = np.array(weights)
        self._dist_weights_mat = np.diag(np.sqrt(self._dist_weights))

    def get_rand_node(self, target_node:Node) -> Node:
        prob = np.random.rand()
        if prob <= self._goal_bias:
            return target_node
        else:
            rand_config = np.random.uniform(self._config_space[:, 0], self._config_space[:, 1])
            return Node(rand_config)

    def get_near_node(self, node:Node) -> Node:
        min_idx = None
        start_time = time.time()
        if self._build_tree_flag:
            self._build_tree_flag = False
            # KDTree with weighted Euclidean distance
            self._kdtree = KDTree(self._configs, metric='minkowski',
                                  p=2, w=self._dist_weights)
            print(f"Built KD tree with nodes {self._kdtree.data.shape[0]} in time {(time.time() - start_time):.4f} s")
            min_idx = self._kdtree.query([node.config], k=1)[1].item()
            return self._elements[min_idx]

        dist_tree, min_idx = self._kdtree.query([node.config], k=1)
        dist_tree, min_idx = dist_tree.item(), min_idx.item()
        if len(self._configs) == self._kdtree.data.shape[0]:
            return self._elements[min_idx]

        diff = np.array(self._configs[self._kdtree.data.shape[0]:]) - node.config # (N, 6)
        dist = np.linalg.norm(diff @ self._dist_weights_mat, axis=1) # (N, )
        min_idx_list = np.argmin(dist)

        if dist[min_idx_list] < dist_tree:
            min_idx = min_idx_list + self._kdtree.data.shape[0]

        # print(f"len {len(self._elements)} | time {(time.time() - start_time):.5f}")
        return self._elements[min_idx]

    def extend_node(self, near_node:Node, rand_node:Node) -> Node:
        unit_vec = rand_node.config - near_node.config
        unit_vec /= np.linalg.norm(unit_vec)
        new_config = near_node.config + unit_vec * self._step_size
        # TODO add comment here
        if np.linalg.norm(new_config - rand_node.config) < self._step_size:
            return Node(rand_node.config)

        new_config = np.clip(new_config, self._config_space[:, 0], self._config_space[:, 1])
        
        return Node(new_config)
    
    def is_connected(self, node_from, node_to):
        if node_from == node_to:
            return True
        
        return False

    def connect_to_target(self, target_node=None):
        # Generate a rand config
        rand_node = self.get_rand_node(target_node)
        # Search for the nearest node
        near_node = self.get_near_node(rand_node)
        prev_node = near_node
        # Connect
        while True:
            new_node = self.extend_node(prev_node, rand_node)
            # Stop if collided
            if self._collision_fn(new_node.config):
                break
            # Stop if fail to extend more
            if new_node == prev_node:
                break

            new_node.parent = prev_node
            self.append(new_node)
            # Stop if connected
            if self.is_connected(new_node, rand_node):
                break

            prev_node = new_node

    def plan_path(self, start_config, goal_config):
        path = []
        # Initialize start and goal node
        start_node = Node(np.array([*start_config]))
        goal_node = Node(np.array([*goal_config]))
        self.append(start_node)
        path_found = False
        start_time = time.time()
        while not path_found:
            # Try connect RRT to goal node
            self.connect_to_target(goal_node)

            # Check if goal node is connected
            if self.is_connected(self.back(), goal_node):
                path_found = True
                break

        if path_found:
            print("Path Found!!!")
            print(f"Time elapsed: {time.time() - start_time:.5f}")
            path = extract_path(self.back())
            draw_sphere_markers(path, BLACK, 2.5)
            draw_sphere_markers(self._configs, BLUE)
            # Path smoothing
            node_path = extract_path(self.back(), keep_node=True)
            node_path = shortcut_smoothing(self, node_path, self._collision_fn)
            path = extract_path(node_path[-1])
            draw_sphere_markers(path, RED)

        return path
