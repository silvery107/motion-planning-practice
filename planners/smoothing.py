import numpy as np
from typing import Callable
import copy

def shortcut_smoothing(rrt_container, 
                       _path:list, 
                       collision_fn:Callable[[list], bool], 
                       max_iterations=150) -> list:
    path = copy.copy(_path)
    for _ in range(max_iterations):
        path_length = len(path)
        assert path_length > 1
        pick_points = np.random.randint(0, path_length, (2,))
        while pick_points[0] == pick_points[1]:
            pick_points = np.random.randint(0, path_length, (2,))

        pick_points.sort()
        first = path[pick_points[0]]
        second = path[pick_points[1]]
        prev_node = first
        success = False
        new_node_list = []
        # Connect
        while True:
            new_node = rrt_container.extend_node(prev_node, second)
            # Stop if collided
            if collision_fn(new_node.config):
                break
            # Stop if fail to extend more
            if new_node == prev_node:
                break

            new_node.parent = prev_node
            new_node_list.append(new_node)
            # Stop if connected
            if rrt_container.is_connected(new_node, second):
                success = True
                break

            prev_node = new_node

        if not success:
            continue

        new_path = []
        new_path.extend(path[:pick_points[0] + 1])
        new_path.extend(new_node_list)
        if pick_points[1] + 1 < path_length:
            path[pick_points[1] + 1].parent = new_node
            new_path.extend(path[pick_points[1] + 1:])
        path = new_path

    print(f"Path smoothed ({len(new_path)} / {len(_path)})")
    return new_path
