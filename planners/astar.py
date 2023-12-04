import time
import numpy as np
from queue import PriorityQueue
from numpy import ndarray
from common import AbcNode
from utils import *


class Node(AbcNode):
    def __init__(self, config:ndarray, parent=None):
        super().__init__(config, parent)

        self.g_cost = float("inf")
        self.h_cost = float("inf")

    def __lt__(self, other: object) -> bool:
        return self.f_cost < other.f_cost

    def get_pos(self):
        return tuple(self.config[:2])

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost


class OpenList:
    def __init__(self) -> None:
        self._queue = PriorityQueue()
        # self._elements = []
        # self._elements = set()
        self._elements = dict()
    
    def __len__(self):
        return self._queue.qsize()

    def put(self, node:Node):
        self._queue.put(node)
        # self._elements.append(node) # list
        # self._elements.add(node) # set
        self._elements[node] = node.g_cost # dict

    def get(self) -> Node:
        node = self._queue.get()
        # self._elements.remove(node) # list
        # self._elements.discard(node) # set

        return node
    
    def empty(self) -> bool:
        return self._queue.empty()
    
    def contains(self, node:Node) -> bool:
        if node in self._elements:
            return True

        return False

    def find_g_cost(self, node:Node) -> float:
        # for item in self._elements:
        #     if node == item:
        #         return item.g_cost
        return self._elements[node]


class ClosedList:
    def __init__(self) -> None:
        self._elements = set()

    def __len__(self):
        return self._elements.__len__()

    def append(self, node:Node):
        self._elements.add(node)

    def contains(self, node:Node) -> bool:
        if node in self._elements:
            return True
            
        return False

def cost(a:Node, b:Node) -> float:
    temp = np.sum(np.square(a.config[:2] - b.config[:2]))
    dtheta = abs(wrap_to_pi(a.config[2] - b.config[2]))

    return np.sqrt(temp + min(dtheta, 2*np.pi - dtheta)**2)

def g_cost(begin:Node, end:Node) -> float:
    return cost(begin, end)

def h_cost(node:Node, goal:Node) -> float:
    # return 0 # A* without h cost is Dijkstra
    return cost(node, goal)


class Astar:
    def __init__(self, config_space, collision_fn, *, connectivity=8, **kwargs):
        self._collison_fn = collision_fn
        self._config_space = config_space
        self._directions = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, np.pi), (0, 0, -np.pi)}
        assert len(self._directions) == 6
        if connectivity == 8:
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    for theta in [-np.pi, 0, np.pi]:
                        self._directions.add((x, y, theta))
            self._directions.remove((0, 0, 0))
            assert len(self._directions) == 26
        self._directions = np.array(list(self._directions))

    def plan_path(self, start_config, goal_config):
        path = []
        start_node = Node(np.array([*start_config]))
        goal_node = Node(np.array([*goal_config]))
        start_node.g_cost = 0.0
        start_node.h_cost = h_cost(start_node, goal_node)
        print("Start:\n", start_node)
        print("Goal:\n", goal_node)
        # Initialize open list and closed list
        open_list = OpenList()
        closed_list = ClosedList()
        path_found = False
        start_time = time.time()
        # Initialize parameters
        iterations = 0
        config_scale = np.array([0.1, 0.1, 0.5])
        exploration_markers = set()
        collision_markers = set()
        # Main search loop
        open_list.put(start_node)
        while not open_list.empty():
            iterations += 1
            # print(f"Iter: {iterations} | Openlist len: {len(open_list)} | Closedlist len: {len(closed_list)}")
            # Process current node with lowest cost
            curr_node = open_list.get()
            # Skip if visited
            if closed_list.contains(curr_node):
                continue

            # Add current node to closed list
            closed_list.append(curr_node)

            # Skip if collided
            if self._collison_fn(curr_node.config):
                collision_markers.add(curr_node.get_pos())
                continue
            
            exploration_markers.add(curr_node.get_pos())

            # Stop if found goal
            if curr_node == goal_node:
                path_found = True
                break

            # Expand node
            for direction in self._directions:
                new_config = curr_node.config + direction * config_scale
                new_config[2] = wrap_to_pi(new_config[2])
                succ_node = Node(new_config)
                # Skip if visited
                if closed_list.contains(succ_node):
                    continue
                # Skip if collided
                # if self._collison_fn((succ_node.x, succ_node.y, succ_node.theta)):
                #     collision_markers.add(succ_node.get_pos())
                #     continue

                temp_g_cost = g_cost(curr_node, succ_node) + curr_node.g_cost
                # Add new successor in open list if it's a new node or a better path
                if not open_list.contains(succ_node) or temp_g_cost < open_list.find_g_cost(succ_node):
                    succ_node.g_cost = temp_g_cost
                    succ_node.h_cost = h_cost(succ_node, goal_node)
                    succ_node.parent = curr_node
                    # This will flush the priority, but a "duplicated config" is pushed to open list,
                    # since we can't update the entire priority queue every time we change its elements.
                    # And it will cause little different if use different underlining lookup container.
                    open_list.put(succ_node)

        print(f"Astar finished with {iterations} iterations")
        print(f"OpenList length {len(open_list)}, ClosedList length {len(closed_list)}")
        if path_found:
            print("Path Found!!!")
            print(f"Time elapsed: {time.time() - start_time:.5f}")
            path = extract_path(curr_node)
            draw_sphere_markers(path, RED)
            draw_sphere_markers(list(exploration_markers), BLUE)

        return path
