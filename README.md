# Motion Planning Practice

Fast motion planning algorithm implementations with a  hovercraft robot demo and a custom maze builder in pybullet!

## Algorithms:
- [x] A*
- [x] RRT
- [x] Bidirectional RRT
- [x] Kinodynamic RRT
- [x] Bidirectional Kinodynamic RRT
- [x] Shortcut Path Smoothing
- [x] BVP Steering
- [x] Fast NN query with lazy-rebuilt KD Tree


## Quick Start
1. Clone this repo
    ```bash
    git clone https://github.com/silvery107/kinodynamic-rrt-pybullet.git
    ```

2. Install required python packages
    ```bash
    chmod +x install.sh
    ./install.sh
    ```

3. Run the demo!
    ```bash
    python demo.py
    ```
    Choose the planning algorithm by setting `--algo`, choices are `["RRT", "BiRRT", "KdRRT", "Astar", "BiKdRRT"]`

#### Known Issues
- BVP solver can be unstable when steering between two states with non-zero velocities