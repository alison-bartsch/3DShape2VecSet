import numpy as np
import open3d as o3d

pcl = np.load('/home/alison/Documents/Apr17_Human_Demos_Difficult_Shapes/pottery/Trajectory0/unnormalized_pointcloud37.npy')

o3d_pcl = o3d.geometry.PointCloud()
o3d_pcl.points = o3d.utility.Vector3dVector(pcl)
o3d.visualization.draw_geometries([o3d_pcl])