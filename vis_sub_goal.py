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
from pcl_animation_gif_generator import generate_colormap, animate_point_cloud, make_gif, make_video
from dist_utils import *

dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
# exp_folder = 'latent_subgoal_more_epochs_smaller_lr_more_augs'  # works okay
# exp_folder = 'latent_subgoal_more_epochs' # FANTASTIC!!!!!
# exp_folder = 'latent_subgoal_more_epochs_even_larger_lr_scheduler' # bad
# exp_folder = 'latent_subgoal_more_epochs_larger_lr_more_augs' # pretty great!!!
# exp_folder = 'latent_subgoal_more_epochs_larger_lr_scheduler' # decent
# exp_folder = 'latent_subgoal_test'  # the worst


model_eval_list = [('latent_subgoal_1_subgoal_steps', 1), 
                   ('latent_subgoal_3_subgoal_steps', 3), 
                   ('latent_subgoal_more_epochs', 5),
                   ('latent_subgoal_7_subgoal_steps', 7)]
# model_eval_list = [('latent_subgoal_3_subgoal_steps', 3)]

for model_elem in tqdm(model_eval_list):
    exp_folder = model_elem[0]

    vis_path = '/home/alison/Documents/GitHub/subgoal_diffusion/subgoal_evals/'
    if not os.path.exists(vis_path + exp_folder):
        os.makedirs(vis_path + exp_folder)


    # define the step size for g.t. visualization
    n_subgoal_steps = model_elem[1] # 1

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
    traj_list = [('/Trajectory0', 33), 
                ('/Trajectory1', 27), 
                ('/Trajectory2', 26), 
                ('/Trajectory3', 26), 
                ('/Trajectory4', 17), 
                ('/Trajectory5', 23)]

    for elem in tqdm(traj_list):
        traj = elem[0]
        traj_path = dataset_dir + traj
        n_states = elem[1]
        goal_idx = n_states - (n_states % n_subgoal_steps) - 1
        og_goal = np.load(traj_path + '/unnormalized_pointcloud' + str(goal_idx) + '.npy')

        if not os.path.exists(vis_path + exp_folder + traj):
            os.makedirs(vis_path + exp_folder + traj)

        for pred_n in range(5):
            pred_folder = '/Run' + str(pred_n) 
            if not os.path.exists(vis_path + exp_folder + traj + pred_folder):
                os.makedirs(vis_path + exp_folder + traj + pred_folder)
            full_save_path = vis_path + exp_folder + traj + pred_folder + '/'

            # iterate through a g.t. trajectory with n_subgoal_steps size
            for i in range(0, goal_idx - n_subgoal_steps, n_subgoal_steps):
                # load in the state
                state_path = traj_path + '/unnormalized_pointcloud' + str(i) + '.npy'
                og_state = np.load(state_path)
                state = np.load(state_path)
                center = np.mean(state, axis=0)
                state -= center
                og_state -= center
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
                # gt_subgoal -= center
                # gt_subgoal = gt_subgoal / np.max(np.abs(gt_subgoal))

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

                # # ground truth next state in green
                # gt_subgoal_pcl = o3d.geometry.PointCloud()
                # gt_subgoal_pcl.points = o3d.utility.Vector3dVector(gt_subgoal)
                # gt_subgoal_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(gt_subgoal, pltmap='summer'))
                # o3d.visualization.draw_geometries([gt_subgoal_pcl])

                # gt_subgoal_list = animate_point_cloud(gt_subgoal, view='isometric', pltmap='summer')
                # print("saving gif...")
                # # make_gif(gt_subgoal_list, filename=full_save_path+'gt_subgoal'+str(i)+'.gif', duration=100)
                # make_video(gt_subgoal_list, filename=full_save_path+'gt_subgoal'+str(i)+'.mp4', fps=15)
                # print("done!")

                np.save(full_save_path + 'gt_subgoal' + str(i) + '.npy', gt_subgoal)

                # # one-step pcl in red
                # one_step_pcl = o3d.geometry.PointCloud()
                # one_step_pcl.points = o3d.utility.Vector3dVector(verts)
                # one_step_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(verts, pltmap='autumn'))
                # o3d.visualization.draw_geometries([one_step_pcl])

                # one_step_subgoal_list = animate_point_cloud(verts, view='isometric', pltmap='autumn')
                # print("saving gif...")
                # # make_gif(one_step_subgoal_list, filename=full_save_path+'one_step_subgoal'+str(i)+'.gif', duration=100)
                # make_video(one_step_subgoal_list, filename=full_save_path+'one_step_subgoal'+str(i)+'.mp4', fps=15)
                # print("done!")

                # unnormalize the verts one_step subgoal
                pred_verts = verts
                verts = verts * np.max(np.abs(og_state)) + center

                np.save(full_save_path + 'one_step_subgoal' + str(i) + '.npy', verts)

                # calculate cd/emd/hd between the gt and predicted subgoal
                dist_metrics = {'CD': chamfer(gt_subgoal, verts),
                                'EMD': emd(gt_subgoal, verts),
                                'HAUSDORFF': hausdorff(gt_subgoal, verts)}
                print("\nSingle-step distance metrics: ", dist_metrics)
                with open(full_save_path + 'single_step_dist_metrics_' + str(i) + '.txt', 'w') as f:
                    f.write(str(dist_metrics))

                # if not first state
                if i != 0:
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

                    # # visualize autoregressive pcl in blue
                    # autoregressive_subgoal_pcl = o3d.geometry.PointCloud()
                    # autoregressive_subgoal_pcl.points = o3d.utility.Vector3dVector(verts)
                    # autoregressive_subgoal_pcl.colors = o3d.utility.Vector3dVector(generate_colormap(verts, pltmap='winter'))
                    # o3d.visualization.draw_geometries([autoregressive_subgoal_pcl])

                    # autoregressive_subgoal_list = animate_point_cloud(verts, view='isometric', pltmap='winter')
                    # print("saving gif...")
                    # # make_gif(autoregressive_subgoal_list, filename=full_save_path+'autoregressive_subgoal'+str(i)+'.gif', duration=100)
                    # make_video(autoregressive_subgoal_list, filename=full_save_path+'autoregressive_subgoal'+str(i)+'.mp4', fps=15)
                    # print("done!")

                    # unnormalize the verts one_step subgoal
                    verts = verts * np.max(np.abs(og_state)) + center
                    
                    np.save(full_save_path + 'autoregressive_subgoal' + str(i) + '.npy', verts)

                    # TODO: calcualte cd/emd between gt and autoregressive subgoal
                    dist_metrics = {'CD': chamfer(gt_subgoal, verts),
                                    'EMD': emd(gt_subgoal, verts),
                                    'HAUSDORFF': hausdorff(gt_subgoal, verts)}
                    print("Autoregressive distance metrics: ", dist_metrics)
                    with open(full_save_path + 'autoregressive_dist_metrics_' + str(i) + '.txt', 'w') as f:
                        f.write(str(dist_metrics))

                # set previous predicted state to verts array downsampled to 2048 points
                idxs = np.random.choice(pred_verts.shape[0], 2048, replace=False)
                prev_state = pred_verts[idxs]
