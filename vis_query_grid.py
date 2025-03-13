import numpy as np
import open3d as o3d

# generate a point cloud of equally spaced points in a 3D cube (total 2000 points)
points = np.random.rand(2000, 3)
points = points * 2 - 1
points = points * 0.5
points = points.astype(np.float32)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])