import numpy as np
from tqdm import tqdm

traj_list = [('/Trajectory0', 33), 
            ('/Trajectory1', 27), 
            ('/Trajectory2', 26), 
            ('/Trajectory3', 26), 
            ('/Trajectory4', 17), 
            ('/Trajectory5', 23)]

data_path = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'

minx = []
miny = []
minz = []
maxx = []
maxy = []
maxz = []
center = []

for traj in tqdm(traj_list):
    traj_path = data_path + traj[0]
    for i in range(traj[1]):
        # load in the point cloud
        pc = np.load(traj_path + f'/unnormalized_pointcloud{i}.npy')
        # get the center of the point cloud
        center.append(np.mean(pc, axis=0))
        pc = pc - np.mean(pc, axis=0)
        # get the min/max x,y,z
        minx.append(np.min(pc[:, 0]))
        miny.append(np.min(pc[:, 1]))
        minz.append(np.min(pc[:, 2]))
        maxx.append(np.max(pc[:, 0]))
        maxy.append(np.max(pc[:, 1]))
        maxz.append(np.max(pc[:, 2]))
        

# calculate the global min/max from the list
print("minx: ", np.min(minx))
print("miny: ", np.min(miny))
print("minz: ", np.min(minz))
print("maxx: ", np.max(maxx))
print("maxy: ", np.max(maxy))
print("maxz: ", np.max(maxz))
print("center: ", np.mean(center, axis=0))

center = np.array([0.628, 0.000, 0.104])