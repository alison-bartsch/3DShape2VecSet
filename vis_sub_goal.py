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
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from pcl_animation_gif_generator import generate_colormap
from dist_utils import *

dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
# exp_folder = 'latent_subgoal_more_epochs_smaller_lr_more_augs'  # works okay
exp_folder = 'latent_subgoal_more_epochs' # FANTASTIC!!!!!
# exp_folder = 'latent_subgoal_more_epochs_even_larger_lr_scheduler' # bad
# exp_folder = 'latent_subgoal_more_epochs_larger_lr_more_augs' # pretty great!!!
# exp_folder = 'latent_subgoal_more_epochs_larger_lr_scheduler' # decent
# exp_folder = 'latent_subgoal_test'  # the worst


# define the step size for g.t. visualization
n_subgoal_steps = 5

# load in the pretrained embedding model
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)

# load in the pretrained subgoal model
model_pth = save_dir + exp_folder + '/best_diffusion_model.pt'
model = models_class_cond.__dict__['kl_d512_m512_l8_edm']()
model.eval()
model.load_state_dict(torch.load(model_pth, map_location='cpu')) #['model'])
model.to(device)

# load in the g.t. goal
# Trajectory0: 33
# Trajectory1: 27
# Trajectory2: 26
# Trajectory3: 26
# Trajectory4: 17
# Trajectory5: 23
traj_path = dataset_dir + '/Trajectory0'
n_states = 33
goal_idx = n_states - (n_states % n_subgoal_steps) - 1
og_goal = np.load(traj_path + '/unnormalized_pointcloud' + str(goal_idx) + '.npy')

# iterate through a g.t. trajectory with n_subgoal_steps size
for i in range(0, goal_idx, n_subgoal_steps):
    # load in the state
    state_path = traj_path + '/unnormalized_pointcloud' + str(i) + '.npy'
    state = np.load(state_path)
    center = np.mean(state, axis=0)
    state -= center
    state = state / np.max(np.abs(state))
    state = torch.from_numpy(state).float()
    state = state.unsqueeze(0)
    state = state.to(device)

    # process the goal point cloud w.r.t. the state center
    goal = og_goal - center
    goal = goal / np.max(np.abs(goal))
    goal = torch.from_numpy(goal).float()
    goal = goal.unsqueeze(0)
    goal = goal.to(device)

    # load in the g.t. subgoal
    gt_subgoal = np.load(traj_path + '/unnormalized_pointcloud' + str(i + n_subgoal_steps) + '.npy')
    gt_subgoal -= center
    gt_subgoal = gt_subgoal / np.max(np.abs(gt_subgoal))

    # create the query grid points   
    density = 64 # 28 # 64 # 45 # 128 original
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)
    print("\n Grid shape: ", grid.shape)

    # predict the subgoal given the real-world previous state
    _, state_latent = ae.encode(state)
    _, goal_latent = ae.encode(goal)

    # sampled_array = model.sample(cond=cond_emb, batch_seeds=torch.arange(i, (i+1)).to(device)).float()
    sampled_array = model.sample(state_cond=state_latent, goal_cond=goal_latent).float()

    print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())
    
    logits = ae.decode(sampled_array[0:1], grid)
    logits = logits.detach()
    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
    verts, faces = mcubes.marching_cubes(volume, 0)
    verts *= gap
    verts -= 1

    m = trimesh.Trimesh(verts, faces)
    # m.show()

    # ground truth next state in green
    gt_subgoal_pcl = o3d.geometry.PointCloud()
    gt_subgoal_pcl.points = o3d.utility.Vector3dVector(gt_subgoal)
    # gt_subgoal_pcl.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]*gt_subgoal.shape[0]))
    gt_subgoal_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(gt_subgoal, pltmap='summer'))
    o3d.visualization.draw_geometries([gt_subgoal_pcl])

    # one-step pcl in red
    print("\nVerts shape: ", verts.shape)
    one_step_pcl = o3d.geometry.PointCloud()
    one_step_pcl.points = o3d.utility.Vector3dVector(verts)
    # one_step_pcl.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]*verts.shape[0]))
    one_step_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(verts, pltmap='autumn'))
    o3d.visualization.draw_geometries([one_step_pcl])

    # calculate cd/emd/hd between the gt and predicted subgoal
    dist_metrics = {'CD': chamfer(gt_subgoal, verts),
                    'EMD': emd(gt_subgoal, verts),
                    'HAUSDORFF': hausdorff(gt_subgoal, verts)}
    print("\nSingle-step distance metrics: ", dist_metrics)

    # if not first state
    print("i: ", i)
    if i != 0:
        print("here!")
        # predict the subgoal given the previous subgoal as the conditioning state
        autoregressive_state = prev_state - center
        autoregressive_state = autoregressive_state / np.max(np.abs(autoregressive_state))
        autoregressive_state = torch.from_numpy(autoregressive_state).float()
        autoregressive_state = autoregressive_state.unsqueeze(0)
        autoregressive_state = autoregressive_state.to(device)
        _, autoregressive_state_latent = ae.encode(autoregressive_state)

        # sampled_array = model.sample(cond=cond_emb, batch_seeds=torch.arange(i, (i+1)).to(device)).float()
        sampled_array = model.sample(state_cond=autoregressive_state_latent, goal_cond=goal_latent).float()

        print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())
        
        logits = ae.decode(sampled_array[0:1], grid)
        logits = logits.detach()
        volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)
        verts *= gap
        verts -= 1

        m = trimesh.Trimesh(verts, faces)
        # m.show()

        # visualize autoregressive pcl in blue
        autoregressive_subgoal_pcl = o3d.geometry.PointCloud()
        autoregressive_subgoal_pcl.points = o3d.utility.Vector3dVector(verts)
        # autoregressive_subgoal_pcl.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]*verts.shape[0]))
        autoregressive_subgoal_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(verts, pltmap='winter'))
        o3d.visualization.draw_geometries([autoregressive_subgoal_pcl])
        # o3d.visualization.draw_geometries([gt_subgoal_pcl, one_step_pcl, autoregressive_subgoal_pcl])

        # TODO: calcualte cd/emd between gt and autoregressive subgoal
        dist_metrics = {'CD': chamfer(gt_subgoal, verts),
                        'EMD': emd(gt_subgoal, verts),
                        'HAUSDORFF': hausdorff(gt_subgoal, verts)}
        print("Autoregressive distance metrics: ", dist_metrics)

    # set previous predicted state to verts array downsampled to 2048 points
    idxs = np.random.choice(verts.shape[0], 2048, replace=False)
    prev_state = verts[idxs]
