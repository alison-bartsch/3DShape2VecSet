import numpy as np
import open3d as o3d
from tqdm import tqdm
from pcl_animation_gif_generator import generate_colormap, animate_point_cloud, make_gif, make_video

step_size_list = [3,5] # [1,3,5,7]

# traj_list = [('/Trajectory0', 32), 
#             ('/Trajectory1', 26), 
#             ('/Trajectory2', 26), 
#             ('/Trajectory3', 26), 
#             ('/Trajectory4', 17), 
#             ('/Trajectory5', 23)]

traj_list = [('/Trajectory2', 26), 
             ('/Trajectory5', 23)]

base_path = '/home/alison/Documents/GitHub/subgoal_diffusion/subgoal_evals/autoregressive_vis/'

for n_subgoal_steps in tqdm(step_size_list):
    model_path = 'latent_subgoal_' + str(n_subgoal_steps) + '_state_idx_global_pcl_normalization_fixed'

    for elem in tqdm(traj_list):
        traj = elem[0]
        traj_path = base_path + model_path + traj
        n_states = elem[1]
        goal_idx = n_states - (n_states % n_subgoal_steps) - 1

        # iterate through a g.t. trajectory with n_subgoal_steps size
        for i in range(0, goal_idx, n_subgoal_steps):
            # load in the gt point cloud
            gt_pcl = np.load(traj_path + '/gt_subgoal' + str(i) + '.npy')

            # generate and save gif to the same directory
            gt_subgoal_list = animate_point_cloud(gt_pcl, view='isometric', pltmap='summer')
            print("saving gif...")
            make_gif(gt_subgoal_list, filename=traj_path+'/gt_subgoal'+str(i)+'.gif', duration=100)
            print("done!")

            # load in the autoregressive point cloud
            autoregressive_pcl = np.load(traj_path + '/autoregressive_subgoal' + str(i) + '.npy')
            # downsample autoregressive point cloud to 2048 points
            # autoregressive_pcl = autoregressive_pcl[np.random.choice(autoregressive_pcl.shape[0], 2048, replace=False), :]

            # generate and save gif to the same directory
            autoregressive_subgoal_list = animate_point_cloud(autoregressive_pcl, view='isometric', pltmap='autumn')
            print("saving gif...")
            make_gif(autoregressive_subgoal_list, filename=traj_path+'/autoregressive_subgoal'+str(i)+'.gif', duration=100)
            print("done!")