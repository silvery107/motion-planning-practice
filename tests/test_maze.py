import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pybullet as p
import time
import pybullet_data

# Connect to PyBullet
p.connect(p.GUI)
p.setGravity(0, 0, -10)

# Load the plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Cell size (larger than the robot dimensions to navigate)
cell_size = 1.
wall_urdf = "assets/maze_wall.urdf"
corner_urdf = "assets/maze_corner.urdf"
boundary_urdf = "assets/maze_boundary.urdf"

# Define the maze layout as a 2D array
maze_layout = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 1, 1, 4, 0, 3, 1, 0],
    [0, 0, 2, 0, 0, 2, 0, 2, 0, 0],
    [0, 0, 6, 1, 1, 5, 0, 6, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 1, 1, 4, 0, 3, 1, 0],
    [0, 0, 2, 0, 0, 2, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Dimensions for the walls based on your robot size (0.5m x 0.3m x 0.15m)
wall_length = 1.0  # slightly larger than the robot length
wall_height = 0.5
wall_thickness = 0.1  # thickness of the walls

# Function to add a wall in PyBullet
def add_wall(x, y, value):
    position = [x * wall_length-5, y * wall_length-5, wall_height/2.]
    if value == 1:  # Horizontal wall
        # Place horizontal wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF(wall_urdf, basePosition=position, baseOrientation=orientation, useFixedBase=True)
    elif value == 2:  # Vertical wall
        # Place vertical wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 1.5707963268])
        p.loadURDF(wall_urdf, basePosition=position, baseOrientation=orientation, useFixedBase=True)
    elif value == 3:  # Top-left corner
        # Place top-left corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, -1.5707963268])
        p.loadURDF(corner_urdf, basePosition=position, baseOrientation=orientation, useFixedBase=True)
    elif value == 4:  # Top-right corner
        # Place top-right corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF(corner_urdf, basePosition=position, baseOrientation=orientation, useFixedBase=True)
    elif value == 5:  # Bottom-left corner
        # Place bottom-left corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 1.5707963268])
        p.loadURDF(corner_urdf, basePosition=position, baseOrientation=orientation, useFixedBase=True)
    elif value == 6:  # Bottom-right corner
        # Place bottom-right corner wall code here
        orientation = p.getQuaternionFromEuler([0, 0, 3.1415926536])
        p.loadURDF(corner_urdf, basePosition=position, baseOrientation=orientation, useFixedBase=True)

p.loadURDF(boundary_urdf, basePosition=[-5, -5, wall_height/2.], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
# Loop through the maze_layout and add walls
for y, row in enumerate(maze_layout):
    for x, cell_value in enumerate(row):
        if cell_value != 0:
            add_wall(x, y, cell_value)

# Run the simulation
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240)

# Cleanup
p.disconnect()
