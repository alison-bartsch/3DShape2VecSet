import os
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import mcubes
import trimesh
import torch
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
import models_ae
import models_class_cond
from real_world_dataset import SubGoalDataset
from action_pred_model import SubGoalValueModel
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from pcl_animation_gif_generator import generate_colormap, animate_point_cloud, make_gif, make_video
from dist_utils import *

def normalize_pcl(pcl):
    min_pos = np.array([-0.058, -0.053, -0.043])
    max_pos = np.array([0.062, 0.062, 0.031])
    # Normalize the point cloud to be between -1 and 1
    pcl = (pcl - min_pos) / (max_pos - min_pos)
    pcl = 2 * pcl - 1
    return pcl

def unnormalize_pcl(pcl):
    min_pos = np.array([-0.058, -0.053, -0.043])
    max_pos = np.array([0.062, 0.062, 0.031])
    # Unnormalize the point cloud from between -1 and 1 to the original range
    pcl = (pcl + 1) / 2
    pcl = pcl * (max_pos - min_pos) + min_pos
    return pcl

# dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
dataset_dir = '/home/alison/Documents/Apr17_Human_Demos_Difficult_Shapes/pottery'
save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
exp_folder = 'latent_subgoal_3_state_idx_global_pcl_normalization_fixed'
value_model_folder = 'subgoal_new_value_model'


vis_path = '/home/alison/Documents/GitHub/subgoal_diffusion/subgoal_evals/autoregressive_vis/'
if not os.path.exists(vis_path + exp_folder):
    os.makedirs(vis_path + exp_folder)


# define the step size for g.t. visualization
n_subgoal_steps = 3 # 3

# load in the pretrained embedding model
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)

# load in the pretrained subgoal model
model_pth = save_dir + exp_folder + '/best_test_loss_diffusion_model.pt' # '/diffusion_model8.pt' 
model = models_class_cond.__dict__['kl_d512_m512_l8_edm']()
model.eval()
model.load_state_dict(torch.load(model_pth, map_location='cpu')) 
model.to(device)

# load in the pretrained value model
value_model_pth = save_dir + value_model_folder + '/best_test_loss_value_model.pt'
value_model = SubGoalValueModel()
value_model.eval()
value_model.load_state_dict(torch.load(value_model_pth, map_location='cpu')) 
value_model.to(device)

# load in the g.t. goal
# traj_list = [('/Trajectory0', 32), 
#             ('/Trajectory1', 26), 
#             ('/Trajectory2', 26), 
#             ('/Trajectory3', 26), 
#             ('/Trajectory4', 17), 
#             ('/Trajectory5', 23)]

traj_list = [('/Trajectory0', 36)]

for elem in tqdm(traj_list):
    traj = elem[0]
    traj_path = dataset_dir + traj
    n_states = elem[1]
    goal_idx = n_states - (n_states % n_subgoal_steps) - 1
    og_goal = np.load(traj_path + '/unnormalized_pointcloud' + str(goal_idx) + '.npy')

    if not os.path.exists(vis_path + exp_folder + traj): # + '_heart'):
        os.makedirs(vis_path + exp_folder + traj) # + '_heart')

    # iterate through a g.t. trajectory with n_subgoal_steps size
    for i in range(0, goal_idx, n_subgoal_steps):
        full_save_path = vis_path + exp_folder + traj + '/' # '_heart' + '/'
        # load in the state
        if i == 0:
            state_path = traj_path + '/unnormalized_pointcloud' + str(i) + '.npy'
            og_state = np.load(state_path)
            state = np.load(state_path)
        else:
            state = best_subgoal[np.random.choice(best_subgoal.shape[0], 2048, replace=False), :]
            og_state = best_subgoal[np.random.choice(best_subgoal.shape[0], 2048, replace=False), :] # TODO: set to best predicted state from previous step
        # center = np.mean(state, axis=0)
        center = np.array([0.628, 0.000, 0.104])
        state -= center
        og_state -= center
        state = normalize_pcl(state)
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0)
        state = state.to(device)

        # process the goal point cloud w.r.t. the state center
        goal = og_goal - center
        goal = normalize_pcl(goal)
        goal = torch.from_numpy(goal).float()
        goal = goal.unsqueeze(0)
        goal = goal.to(device)

        # visualize the conditioning state and goal info
        state_pcl = o3d.geometry.PointCloud()
        state_pcl.points = o3d.utility.Vector3dVector(state.cpu().numpy()[0])
        state_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(state.cpu().numpy()[0], pltmap='summer'))
        goal_pcl = o3d.geometry.PointCloud()
        goal_pcl.points = o3d.utility.Vector3dVector(goal.cpu().numpy()[0])
        goal_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(goal.cpu().numpy()[0], pltmap='autumn'))
        # o3d.visualization.draw_geometries([state_pcl])
        # o3d.visualization.draw_geometries([goal_pcl])

        # load in the g.t. subgoal
        gt_subgoal = np.load(traj_path + '/unnormalized_pointcloud' + str(i + n_subgoal_steps) + '.npy')

        # create the query grid points   
        density = 64 # 28 # 64 # 45 # 128 original
        gap = 2. / density
        x = np.linspace(-1, 1, density+1)
        y = np.linspace(-1, 1, density+1)
        z = np.linspace(-1, 1, density+1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

        # iterate to generate population of candidate subgoals
        subgoal_candidates = {} # dictionary to hold the subgoal candidates with their corresponding values
        # for j in range(16):
        for j in range(2):
            # predict the subgoal given the real-world previous state
            _, state_latent = ae.encode(state)
            _, goal_latent = ae.encode(goal)

            print("State index: ", torch.tensor(i).unsqueeze(0).unsqueeze(0).float())
            sampled_array = model.sample(state_cond=state_latent, goal_cond=goal_latent, state_idx=torch.tensor(i).unsqueeze(0).unsqueeze(0).float().to(device)).float()
            
            logits = ae.decode(sampled_array[0:1], grid)
            logits = logits.detach()
            volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
            verts, faces = mcubes.marching_cubes(volume, 0)
            verts *= gap
            verts -= 1

            # downsample verts
            print("verts shape: ", verts.shape)
            if verts.shape[0] > 2048:
                downsampled_verts = verts[np.random.choice(verts.shape[0], 2048, replace=False), :]
            else:
                downsampled_verts = verts[np.random.choice(verts.shape[0], 2048, replace=True), :]
            # convert verts to correct torch format
            downsampled_verts = torch.from_numpy(downsampled_verts).unsqueeze(0).float()
            downsampled_verts = downsampled_verts.to(device)
            # pass verts through ae encoder
            _, latent_verts = ae.encode(downsampled_verts)
            # pass state_latent, goal_latent and verts through value model
            value = value_model(state_latent, goal_latent, latent_verts)
            value = value.detach().cpu().numpy()
            print("subgoal candidate value: ", value)

            # calculate emd between unnormalized verts and g.t. subgoal
            unnorm_verts = unnormalize_pcl(verts) + center
            emd_dist = emd(unnorm_verts, gt_subgoal)
            print("emd distance: ", emd_dist)

            # add j, value and verts (not downsampled) to subgoal_candidates
            subgoal_candidates[j] = {'value': value, 'emd':emd_dist, 'verts': verts}

        # select the best subgoal candidate (i.e. the one with the smallest value)
        best_subgoal_idx = min(subgoal_candidates, key=lambda x: subgoal_candidates[x]['emd'])
        # best_subgoal_idx = min(subgoal_candidates, key=lambda x: subgoal_candidates[x]['value'])
        print("best subgoal candidate: ", best_subgoal_idx)
        # print("best subgoal value: ", subgoal_candidates[best_subgoal_idx]['value'])
        print("best subgoal emd: ", subgoal_candidates[best_subgoal_idx]['emd'])
        best_subgoal = subgoal_candidates[best_subgoal_idx]['verts']
        # unnormalize the best subgoal candidate
        # best_subgoal = best_subgoal * np.max(np.abs(og_state)) + center
        best_subgoal = unnormalize_pcl(best_subgoal) + center

        # check if all verts in subgoal_candidates dictionary are the same
        for k in range(16):
            if not np.array_equal(subgoal_candidates[best_subgoal_idx]['verts'], subgoal_candidates[k]['verts']):
                print("subgoal candidates are not the same!")
                break
            else:
                print("subgoal candidates are the same!")

        # ground truth next state in green
        gt_subgoal_pcl = o3d.geometry.PointCloud()
        gt_subgoal_pcl.points = o3d.utility.Vector3dVector(gt_subgoal)
        gt_subgoal_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(gt_subgoal, pltmap='summer'))
        o3d.visualization.draw_geometries([gt_subgoal_pcl])

        # gt_subgoal_list = animate_point_cloud(gt_subgoal, view='isometric', pltmap='summer')
        # print("saving gif...")
        # # make_gif(gt_subgoal_list, filename=full_save_path+'gt_subgoal'+str(i)+'.gif', duration=100)
        # make_video(gt_subgoal_list, filename=full_save_path+'gt_subgoal'+str(i)+'.mp4', fps=15)
        # print("done!")

        # one-step pcl in red
        one_step_pcl = o3d.geometry.PointCloud()
        one_step_pcl.points = o3d.utility.Vector3dVector(best_subgoal)
        one_step_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(best_subgoal, pltmap='autumn'))
        o3d.visualization.draw_geometries([one_step_pcl])

        # autoregressive_subgoal_list = animate_point_cloud(best_subgoal, view='isometric', pltmap='autumn')
        # print("saving gif...")
        # # make_gif(one_step_subgoal_list, filename=full_save_path+'one_step_subgoal'+str(i)+'.gif', duration=100)
        # make_video(one_step_subgoal_list, filename=full_save_path+'one_step_subgoal'+str(i)+'.mp4', fps=15)
        # print("done!")

        # calculate cd/emd/hd between the gt and predicted subgoal
        dist_metrics = {'CD': chamfer(gt_subgoal, best_subgoal),
                        'EMD': emd(gt_subgoal, best_subgoal),
                        'HAUSDORFF': hausdorff(gt_subgoal, best_subgoal)}
        print("\nSingle-step distance metrics: ", dist_metrics)
        with open(full_save_path + 'single_step_dist_metrics_' + str(i) + '.txt', 'w') as f:
            f.write(str(dist_metrics))

        # save the point clouds
        np.save(full_save_path + 'gt_subgoal' + str(i) + '.npy', gt_subgoal)
        np.save(full_save_path + 'autoregressive_subgoal' + str(i) + '.npy', best_subgoal)