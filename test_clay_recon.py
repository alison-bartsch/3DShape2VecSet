import os
import numpy as np
import open3d as o3d
import mcubes
import trimesh
import torch
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
import models_ae
from action_pred_model import ActionPredModel
from real_world_dataset import ActionPredDataset

dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
# save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
# exp_folder = 'latent_action_pcl_normalized_smaller_lr_scheduler' 
# os.makedirs(save_dir + exp_folder)

# create datasets and dataloaders for train/test
train_dataset = ActionPredDataset(dataset_dir, [0,1,2,3], 5)
test_dataset = ActionPredDataset(dataset_dir, [4,5], 5)
train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# load in the pretrained embedding model
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)

# iterate through the train_loader to embed and reconstruct the point clouds
ae.eval()
for state1, state2, _ in tqdm(train_loader):
    # keep the original point clouds
    s1_pcl = state1.squeeze(0).cpu().numpy()
    s2_pcl = state2.squeeze(0).cpu().numpy()

    # embed the point clouds
    state1 = state1.to(device)
    state2 = state2.to(device)
    latent1 = ae.encode(state1)[1]
    latent2 = ae.encode(state2)[1]

    # # create the query grid points
    density = 45 # 128 original
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)
    print("\n Grid shape: ", grid.shape)

    # # # create the query grid points
    # # # 1024 points in bounded volume [-1,1]^3
    # query_pts = torch.rand(1024, 3, device=device) * 2 - 1
    # grid = query_pts.unsqueeze(0)

    # # get 1024 points from near-surface region of the state
    
    # visualize query_pts with open3d
    query_pts = grid.squeeze(0).cpu().numpy()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(query_pts)
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]*query_pts.shape[0]))

    # visualize the original point clouds
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(s1_pcl)
    pcd2.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]*s1_pcl.shape[0]))
    # o3d.visualization.draw_geometries([pcd1, pcd2])

    # decode the point clouds
    logits = ae.decode(latent1, grid)

    logits = logits.detach()
    
    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
    verts, faces = mcubes.marching_cubes(volume, 0)

    verts *= gap
    verts -= 1

    m = trimesh.Trimesh(verts, faces)
    m.show()


# density = 128
# gap = 2. / density
# x = np.linspace(-1, 1, density+1)
# y = np.linspace(-1, 1, density+1)
# z = np.linspace(-1, 1, density+1)
# xv, yv, zv = np.meshgrid(x, y, z)
# grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

# total = 1000
# iters = 100


# with torch.no_grad():
#     for category_id in [18]:
#         print(category_id)
#         for i in range(1000//iters):
#             sampled_array = model.sample(cond=torch.Tensor([category_id]*iters).long().to(device), batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()

#             print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

#             for j in range(sampled_array.shape[0]):
                
#                 logits = ae.decode(sampled_array[j:j+1], grid)

#                 logits = logits.detach()
                
#                 volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
#                 verts, faces = mcubes.marching_cubes(volume, 0)

#                 verts *= gap
#                 verts -= 1

#                 m = trimesh.Trimesh(verts, faces)
#                 m.export('class_cond_obj/{}/{:02d}-{:05d}.obj'.format('kl_d512_m512_l16_edm', category_id, i*iters+j))