import argparse
import datetime
import json
import os
import time
import numpy as np
import open3d as o3d
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(8)

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_shape_surface_occupancy_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_ae

from engine_ae import train_one_epoch, evaluate

# ae_pth = '/home/alison/Downloads/pretrained/class_encoder_55_512_1024_24_K1024_vqvae_512_1024_2048/checkpoint-399.pth' # TODO: add in the path to pretrained ae weights
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
print("Loading autoencoder %s" % ae_pth)
# ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)

pcl = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory5/unnormalized_pointcloud9.npy')
pcl = torch.from_numpy(pcl).to(device).unsqueeze(0).float()
print("pcl shape: ", pcl.shape)

# encode the point cloud
latent = ae.encode(pcl)
print("\nLatent: ", latent)

# print the latent shape
print("\nLatent pcl shape: ", latent[1].shape)



# # decode the point cloud
# ae.decode(latent, queries)

