import numpy as np
from sklearn.neighbors import KDTree, BallTree
import time

# Generate random sample data
np.random.seed(0)
data = np.random.rand(10000, 6)  # 10,000 points in 6 dimensions

# Generate random query points
query_points = np.random.rand(100, 6)  # 100 points in 6 dimensions

# Benchmark KDTree
start_time = time.time()
kd_tree = KDTree(data, metric='minkowski', p=2)
kd_tree_construct_time = time.time() - start_time

start_time = time.time()
kd_tree.query(query_points, k=1)
kd_tree_query_time = time.time() - start_time

# Benchmark BallTree
start_time = time.time()
ball_tree = BallTree(data)
ball_tree_construct_time = time.time() - start_time

start_time = time.time()
ball_tree.query(query_points, k=1)
ball_tree_query_time = time.time() - start_time

# Output the benchmark results
print(f"KDTree construction time: {kd_tree_construct_time:.4f} seconds")
print(f"KDTree query time: {kd_tree_query_time:.4f} seconds")
print(f"BallTree construction time: {ball_tree_construct_time:.4f} seconds")
print(f"BallTree query time: {ball_tree_query_time:.4f} seconds")
print(kd_tree.data.shape)
