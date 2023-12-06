# Motion Planning Practice

Fast motion planning algorithm implementations with a  hovercraft robot demo and a custom maze builder in pybullet!

<img src="figures/KdRRT.png" width="450">

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
    python demo.py # default to Kinodynamic RRT
    ```
    Choose a different planning algorithm by setting `--algo`, choices are `["RRT", "BiRRT", "KdRRT", "Astar", "BiKdRRT"]`.

4. Customize your maze. Just edit the maze matrix in `maze_layout.json` and have fun.
   Make sure the goal (shown in green) is feasible.

## Gallery

<p align="left">
<img src="figures/BiKdRRT.png" width="300">
<img src="figures/BiRRT.png" width="300">
</p>
<p align="left">
<img src="figures/RRT.png" width="300">
<img src="figures/Astar.png" width="300">
</p>


#### Known Issues
- BVP solver can be unstable when steering between two states with large velocities